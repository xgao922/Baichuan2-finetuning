import os
import json
import re
from tqdm import tqdm


def process_spaces(chinese_text):
    # 使用正则表达式匹配空格
    # space_pattern = re.compile(r'(?<=[\u4e00-\u9fff0-9])\s+|\s+(?=[\u4e00-\u9fff0-9])')

    space_pattern = re.compile(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])\s+(?=\d)|(?<=\d)\s+(?=[\u4e00-\u9fff])|(?<=[a-zA-Z])\s+(?=[\u4e00-\u9fff])')

    # 替换文本中的空格
    processed_text = space_pattern.sub('', chinese_text)

    return processed_text


# text = "玛 丽   兰 姆 玛 丽 me 安  兰 姆 Mary Ann Lamb  是 一 位 英 格 兰 作 家 2 0 0  查 尔 斯  兰 姆 的 姐 姐  "
# filtered_text = process_spaces(text)
# print(filtered_text)


def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        # 循环遍历并移除空格行
        lines = [line.strip() for line in lines if line.strip()]

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            # 去除特殊字符和标点符号
            line = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9\s]+", " ", line)
            # print(line)

            # 去除空格
            line = re.sub(r'\s+', ' ', line)
            line = process_spaces(line)
            line = line.lstrip() # 删除每行开头空格

            output_file.write(line + '\n')


def process_files_in_folder(input_folder, output_folder):
    input_file_names = sorted(os.listdir(input_folder), key=lambda x: int(x.split('-')[0]))

    for input_filename in tqdm(input_file_names, desc="Processing Files", unit="file"):
        input_file_path = os.path.join(input_folder, input_filename)
        if os.path.isfile(input_file_path):
            relative_path = os.path.relpath(input_file_path, input_folder)
            output_file_path = os.path.join(output_folder, relative_path)

            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            process_file(input_file_path, output_file_path)


input_folder_path = "/home/xgao/Baichuan2/fine-tune/data/0.1_0.03_931"
output_folder_path = "/home/xgao/Baichuan2/fine-tune/data/filtered_corpus"

process_files_in_folder(input_folder_path, output_folder_path)


