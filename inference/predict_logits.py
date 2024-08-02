from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 使用固定的随机种子

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def measure_hit_ratio(num_outputs, data_path, model_path, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    with open(data_path, 'r') as file:
        samples = json.load(file)
        num_samples = len(samples)
        # print(samples)
        # print(samples[0])
        # print(num_samples)

        count = 0

        # 创建一个tqdm进度条
        for sample in tqdm(samples, desc="Precessing"):

            # 获取prefix和gold字段的值
            input_text = sample["prefix"]
            label = sample["gold"]

            # # 打印prefix和gold
            # print(f'input_text: {input_text}')
            # print(f'label: {label}')

            input_tokens = tokenizer(input_text, return_tensors='pt')
            model_inputs = input_tokens.to('cuda:1')

            # 获取logits
            with torch.no_grad():
                logits = model(**model_inputs).logits[:, -1, :]

            next_token = torch.argmax(logits, dim=-1).reshape(-1)[0]
            # print(f"next_token:{next_token}")

            next_word = tokenizer.decode(next_token)
            # print(f"next_word:{next_word}")

            # 获取前k个最高的logits对应的词汇索引
            # for num in [5, 10, 20, 50, 100]:
            topk_indices = torch.topk(logits, k=num_outputs, dim=-1).indices
            # print("topk_indices:", topk_indices)
            
            topk_word = []

            # 将索引转换为对应的词汇
            for idx in topk_indices[0, :]:
                next = tokenizer.decode(idx.item(), skip_special_tokens=True)
                topk_word.append(next)
                # print(next)

            for word in topk_word:
                if label == word[0]:
                    count += 1
                    break

            # 输出分割线
            # print("-" * 40)

            with open(output_file, 'a') as output_f:
                output_f.write(f'input_text: {input_text}\n')
                output_f.write(f'label: {label}\n')
                output_f.write(f'topk_word: {topk_word}')
                output_f.write("-" * 40 + '\n')

    hit_ratio = count / num_samples
    print(f"Hit ratio for num_outputs={num_outputs}: {hit_ratio}")
    return hit_ratio

if __name__ == "__main__":
    import sys
    num_outputs = int(sys.argv[1])
    model_path = sys.argv[2]
    # data_path = '/data/xgao/Baichuan2/test_set_400.json' # test_set_400.json
    data_path = '/home/xgao/Baichuan2/fine-tune/samples.json'
    output_file = 'predict.txt'
    measure_hit_ratio(num_outputs, data_path, model_path, output_file)