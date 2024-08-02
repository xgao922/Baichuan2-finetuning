import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json
from typing import Dict

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
from transformers import AutoTokenizer

from peft import AutoPeftModelForCausalLM

import time
import pdb
import re


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")


@dataclass
class DataArguments:
    data_folder: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


def separate_chinese_english(text):
    chinese_text = re.sub(r'[^\u4e00-\u9fff]+', '+', text)  # 匹配非中文字符，替换为空格
    english_text = re.sub(r'[\u4e00-\u9fff]+', '-', text)  # 匹配中文字符，替换为空格
    # print("chinese_text:", chinese_text)
    # print("english_text:", english_text)

    return chinese_text, english_text


class UnsupervisedDataset(Dataset):
    """Dataset for unsupervised fine-tuning."""

    def __init__(
            self,
            data_folder,
            model_max_length,
            tokenizer,
    ):
        super(UnsupervisedDataset, self).__init__()
        self.data_folder = data_folder
        self.model_max_length = model_max_length
        self.tokenizer = tokenizer

        # 获取文件夹中的所有文件
        self.file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if
                           os.path.isfile(os.path.join(data_folder, file))]

    def __len__(self):
        return len(self.file_paths)

    def preprocessing(self, text):

        text = str(text)
        # print(f"text: {text}")

        # 将文本分成小段
        segments = [text[i:i + self.model_max_length] for i in range(0, len(text), self.model_max_length)]

        input_ids_list = []
        labels_list = []
        attention_masks_list = []

        for segment in segments:
            # print("segment", segment)
            # 使用分词器对文本进行编码
            # input_ids = self.tokenizer.encode(segment, padding="max_length", max_length=self.model_max_length)
            # print("segment:", segment)
            chinese_text, english_text = separate_chinese_english(segment)
            chinese_tokens = [char for char in chinese_text]
            english_tokens = self.tokenizer.tokenize(english_text)
            # print("chinese_tokens:", chinese_tokens)
            # print("english_tokens:", english_tokens)

            input_tokens = []
            count = 0
            if english_tokens[0] == '-':
                for zh_char in chinese_tokens:
                    if zh_char != '+':
                        input_tokens.append(zh_char)
                    else:
                        count += 1
                        while count < len(english_tokens) and english_tokens[count] != '-':
                            input_tokens.append(english_tokens[count])
                            count += 1
            elif chinese_tokens[0] == '+':
                for en_char in english_tokens:
                    if en_char != '-':
                        input_tokens.append(en_char)
                    else:
                        count += 1
                        while count < len(chinese_tokens) and chinese_tokens[count] != '+':
                            input_tokens.append(chinese_tokens[count])
                            count += 1

            # print("input_tokens:", input_tokens)
            # print(len(input_tokens))

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            input_ids[len(input_tokens):] = [0] * (self.model_max_length - len(input_tokens))
            # print("input_ids:", input_ids)

            # 将标签后移一位
            labels = input_ids[:]
            # labels = input_ids[1:] + [self.tokenizer.pad_token_id]
            # print("labels:", labels)

            # 创建注意力掩码
            attention_mask = [1] * len(input_ids)
            attention_mask[len(input_tokens):] = [0] * (self.model_max_length - len(input_tokens))
            # print("attention_mask:", attention_mask)
            # time.sleep(10)

            input_ids_tensor = torch.LongTensor(input_ids)
            labels_tensor = torch.LongTensor(labels)
            attention_masks_tensor = torch.LongTensor(attention_mask)

            # print("input_ids_tensor", input_ids_tensor)
            # print(input_ids_tensor.shape)
            # time.sleep(10)

        # 返回预处理后的结果
        yield {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_masks_tensor,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 读取长文本数据文件
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', ' ')
            # print(f"text: {text}")

        # 调用 preprocessing 方法，迭代返回每个 segment 的信息
        for segment_info in self.preprocessing(text):
            return segment_info


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    # model = AutoPeftModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        model.enable_input_require_grads()

        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    # def log_metrics(metrics, step):
    #     for key, value in metrics.item():
    #         writer.add_scalar(key, value, step)

    dataset = UnsupervisedDataset(
        data_args.data_folder, training_args.model_max_length, tokenizer
    )

    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )

    trainer.train()

    # 保存训练器的状态
    trainer.save_state()

    # 将训练好的模型保存到指定的目录
    trainer.save_model(output_dir=training_args.output_dir)

    # writer.close()


# Python的标准模式，确保代码作为主程序运行时才执行下面的内容
if __name__ == "__main__":
    # from torch.utils.tensorboard import SummaryWriter

    # writer = SummaryWriter(log_dir="tensorboard")

    # 调用上面定义的train函数，开始训练过程
    train()
