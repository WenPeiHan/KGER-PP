#-*-coding -utf-8 -*-
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
random.seed(42)
#
# model_path = "D:\Code\PrML\Bert-base-uncased"
#
# model_path = "D:\Code\PrML\Bert-base-Chinese"

# model_path = "D:\Code\PrML\google-flan-t5-base"

model_path = "/root/autodl-tmp/PrML/flan-t5-base"

data_path = "../../DataSet/data/data_en.json"

def read_json(dir):
    # 打开JSON文件并读取内容
    with open(dir, 'r', encoding='utf-8') as json_file:
        file_content = json_file.read()

    # 解析JSON字符串为字典
    data_dict = json.loads(file_content)
    return data_dict

data = read_json(data_path)

'''
train_data, test_data = train_test_split(data, test_size=0.6, random_state=42)

classified = defaultdict(list)
specific_key = "questions_template"

for item in data:
    # Check if the key exists in the dictionary
    if specific_key in item:
        classified[item[specific_key]].append(item)

data_classified = dict(classified)

data_balance = []

for key, value_list in data_classified.items():

    value_list = [item for item in value_list if item["rationale_use"] == 1]

    if len(value_list) > 30:
        value_list = random.sample(value_list, 30)

    if len(value_list) < 10:
        continue

    data_balance.extend(value_list)

classified_b = defaultdict(list)

for item2 in data_balance:
    # Check if the key exists in the dictionary
    if specific_key in item2:
        classified_b[item2[specific_key]].append(item2)

data_classified_b = dict(classified_b)

data = data_balance
'''

classified = defaultdict(list)
specific_key = "questions_template"

for item in data:
    # Check if the key exists in the dictionary
    if specific_key in item:
        classified[item[specific_key]].append(item)

data_classified = dict(classified)
data_balance = []

for key, value_list in data_classified.items():
    value_list = [item for item in value_list if item["rationale_use"] == 1]
    if len(value_list) > 30:
        value_list = random.sample(value_list, 30)
    if len(value_list) < 10:
        continue
    data_balance.extend(value_list)

classified_b = defaultdict(list)
for item2 in data_balance:
    # Check if the key exists in the dictionary
    if specific_key in item2:
        classified_b[item2[specific_key]].append(item2)

data_classified_b = dict(classified_b)
train_data, test_data = [], []

for key_b, value_list_b in data_classified_b.items():
    train_data_set, test_data_set = train_test_split(value_list_b, test_size=0.1, random_state=42)
    train_data.extend(train_data_set)
    test_data.extend(test_data_set)

prompt_list = []
target_list = []

for item in train_data:
    # if item["type"] == 1:
    #     question = item["question_e"]
    #     kopl = item["kopl_e"]
    #     rationale = item["rationale_e"]
    #
    #     prompt = "问题是：{}".format(question) + "推理路径是：{}".format(kopl)
    #     target = "解释：{}".format(rationale)
    #
    #     prompt_list.append(prompt)
    #     target_list.append(target)

    # if item["type"] == 1:
    #     question = data["question"]
    #     kopl = data["kopl"]
    #     triple = data["triple_text"]
    #     triple_text = ";".join(triple)
    #
    #     rationale = data["rationale"]
    #     answer = data["answer"]
    #     answer = ";".join(map(str, answer))
    #
    #     prompt = "Question: {}".format(question) + " Triple: {}".format(triple_text)
    #     target = "Rationale：{}".format(rationale) + " Answer:{}".format(answer)
    #
    #     prompt_list.append(prompt)
    #     target_list.append(target)
    #
    # if item["type"] == 2:
    #     question = data["question"]
    #     kopl = data["kopl"]
    #     triple = data["triple_text"]
    #     triple_text = ";".join(triple)
    #
    #     rationale = data["rationale"]
    #     answer = data["answer"]
    #     answer = ";".join(map(str, answer))
    #
    #     prompt = "Question: {}".format(question) + " Triple: {}".format(triple_text)
    #     target = "Rationale：{}".format(rationale) + " Answer:{}".format(answer)
    #
    #     prompt_list.append(prompt)
    #     target_list.append(target)


    question = item["question"]
    kopl = item["kopl"]
    triple = item["triple_text"]
    if len(triple) > 10:
        triple = triple[:10]
    triple_text = ";".join(triple)

    rationale = item["rationale"]
    answer = item["answer"]

    if len(answer) > 10:
        answer = answer[:10]

    answer = ";".join(map(str, answer))

    # prompt = "Question: {}".format(question) + " Triple: {}".format(triple_text)
    prompt = "Question: {}".format(question) + " Reasoning Path: {}".format(kopl)
    # target = "Rationale：{}".format(rationale) + " Answer:{}".format(answer)
    target = "Rationale：{}".format(rationale)

    prompt_list.append(prompt)
    target_list.append(target)

# kopl_list = [item["kopl"] for item in data if "kopl" in item]
# rationale_list = [item["rationale"] for item in data if "rationale" in item]

# kopl_list = [item["kopl"] for item in train_data if "kopl" in item and item["type"] == 2]
# rationale_list = [item["rationale"] for item in train_data if "rationale" in item and item["type"] == 2]

kopl_list = prompt_list
rationale_list = target_list

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

'''
# 示例数据集，这里用简单的列表代替
source_texts = ["This is a sample sentence.", "Another example sentence here."]
target_texts = ["Output sentence 1", "Output sentence 2"]

# 统计源文本的词元长度
source_lengths = [len(tokenizer.encode(text)) for text in source_texts]
# 统计目标文本的词元长度
target_lengths = [len(tokenizer.encode(text)) for text in target_texts]'''

# 统计源文本的词元长度
source_lengths = [len(tokenizer.encode(text)) for text in kopl_list]
# 统计目标文本的词元长度
target_lengths = [len(tokenizer.encode(text)) for text in rationale_list]

# 打印统计信息
print(f"Source text length statistics: Mean={np.mean(source_lengths)}, Median={np.median(source_lengths)}, Max={np.max(source_lengths)}")
print(f"Target text length statistics: Mean={np.mean(target_lengths)}, Median={np.median(target_lengths)}, Max={np.max(target_lengths)}")

# 绘制直方图（可选）
plt.hist(source_lengths, bins=20, alpha=0.5, label='Source lengths')
plt.hist(target_lengths, bins=20, alpha=0.5, label='Target lengths')
plt.xlabel('Token length')
plt.ylabel('Frequency')
plt.legend()
plt.show()