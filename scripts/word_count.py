import os

folder_path = "/home/xgao/Baichuan2/fine-tune/data/test_set"

file_names = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

for filename in file_names:
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            lines = file.readlines()
            print(f"{filename}的总词数为{len(content)}")