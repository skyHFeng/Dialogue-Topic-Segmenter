import argparse
import random
from transformers import BertTokenizer
import re


""" 加载对话的句子内容 """
def load_txt(in_fname):
    id2txt = {}
    with open(in_fname, encoding='utf-8') as in_file:
        for idx, line in enumerate(in_file):
            id2txt[idx] = [utterance.replace(" __eou__", "") for utterance in line.strip().split(" __eou__ ")]
    return id2txt


""" 加载对话的句子行为 """
def load_act(in_fname):
    id2act = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            id2act[idx] = line.strip().split(" ")
    return id2act


""" 加载对话的主题 """
def load_topic(in_fname):
    id2topic = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            id2topic[idx] = line.strip()
    return id2topic


text_path = 'data/ijcnlp_dailydialog/dialogues_text.txt'
topic_path = 'data/ijcnlp_dailydialog/dialogues_topic.txt'
act_path = 'data/ijcnlp_dailydialog/dialogues_act.txt'

# load all the dialogues and their features...
txt_dict = load_txt(text_path)
topic_dict = load_topic(topic_path)
act_dict = load_act(act_path)


# extract the utterance pairs with patterns: 2-1, 3-4 （按 2-1, 3-4 的句子行为构造话语对）
# The dialog act number represents: { 1: inform，2: question, 3: directive, 4: commissive }
tuples = []; win_size = 1; count_no = 0

""" 遍历对话 """
for idx in range(13118):
    # 第673段对话疑似有问题，跳过
    if idx == 672:
        continue

    # 当前对话的句子内容
    utterances = txt_dict[idx]
    # 当前对话的句子行为
    acts = act_dict[idx]
    # 当前对话的主题
    topic = topic_dict[idx]

    # 根据对话的句子行为，遍历当前对话的句子
    for a_idx in range(len(acts)-1):
        # 若当前句子的行为为 ‘2: question’
        if acts[a_idx] == '2':
            # 若下一个句子的行为为‘1: inform’
            if acts[a_idx+1] == '1':
                # 则，相邻两个句子构成正样本
                positive_sample = [utterances[a_idx], utterances[a_idx+1]]
                # 同时，获取当前对话句子行为不为‘1: inform’的句子列表
                utterances_wo_1 = [utterances[i] for i in range(len(utterances)) if acts[i] != '1']
                # 若 utterances_wo_1 的长度大于1，即除当前句子外还有其它句子
                try:
                    # utterances_wo_1 中，当前句子位于起始位置
                    if a_idx-1-win_size < 0:
                        # 当前句子与同一对话中之后不相邻的句子构成第一种负样本
                        negative_sample_1 = [utterances[a_idx], random.choice(utterances_wo_1[a_idx+1+win_size:])]
                    # utterances_wo_1 中，当前句子位于中间位置
                    else:
                        # 当前句子与同一对话中之前或之后不相邻的句子构成第一种负样本
                        negative_sample_1 = [utterances[a_idx], random.choice(utterances_wo_1[:a_idx-win_size]+utterances_wo_1[a_idx+1+win_size:])]
                # 否则，没有第一种负样本
                except:
                    # print('there is no negative sample 1...')
                    count_no += 1
                    negative_sample_1 = []
                # 从所有对话中，随计选择一个与当前对话不同主题的对话
                sampled_dial = txt_dict[random.choice([key for key, value in topic_dict.items() if value != topic])]
                # 从选择的不同主题的对话中，随机选择一个句子，与当前句子构成第二种负样本
                negative_sample_2 = [utterances[a_idx], random.choice(sampled_dial)]

                # 若当前句子没有第一种负样本，只收集正样本和第二种负样本
                if negative_sample_1 == []:
                    tmp = [positive_sample, negative_sample_2]
                    tuples.append(tmp)
                # 否则，收集正样本、第一种负样本和第二种负样本
                else:
                    tmp = [positive_sample, negative_sample_1, negative_sample_2]
                    tuples.append(tmp)

        # 若当前句子的行为为‘3: directive’
        if acts[a_idx] == '3':
            # 若下一个句子的行为为‘4: commissive’
            if acts[a_idx+1] == '4':
                # 则，相邻两个句子构成正样本
                positive_sample = [utterances[a_idx], utterances[a_idx+1]]
                # 同时，获取当前对话句子行为不为‘4: commissive’的句子列表
                utterances_wo_4 = [utterances[i] for i in range(len(utterances)) if acts[i] != '4']
                # 若 utterances_wo_4 的长度大于1，即除当前句子外还有其它句子
                try:
                    # utterances_wo_4 中，当前句子位于起始位置
                    if a_idx-1-win_size < 0:
                        # 当前句子与同一对话中之后不相邻的句子构成第一种负样本
                        negative_sample_1 = [utterances[a_idx], random.choice(utterances_wo_4[a_idx+1+win_size:])]
                    # utterances_wo_1 中，当前句子位于中间位置
                    else:
                        # 当前句子与同一对话中之前或之后不相邻的句子构成第一种负样本
                        negative_sample_1 = [utterances[a_idx], random.choice(utterances_wo_4[:a_idx-win_size]+utterances_wo_4[a_idx+1+win_size:])]
                # 否则，没有第一种负样本
                except:
                    # print('there is no negative sample 1...')
                    count_no += 1
                    negative_sample_1 = []
                # 从所有对话中，随计选择一个与当前对话不同主题的对话
                sampled_dial = txt_dict[random.choice([key for key, value in topic_dict.items() if value != topic])]
                # 从选择的不同主题的对话中，随机选择一个句子，与当前句子构成第二种负样本
                negative_sample_2 = [utterances[a_idx], random.choice(sampled_dial)]

                # 若当前句子没有第一种负样本，只收集正样本和第二种负样本
                if negative_sample_1 == []:
                    tmp = [positive_sample, negative_sample_2]
                    tuples.append(tmp)
                # 否则，收集正样本、第一种负样本和第二种负样本
                else:
                    tmp = [positive_sample, negative_sample_1, negative_sample_2]
                    tuples.append(tmp)
    # print(idx)
print(len(tuples))
print(count_no)


# 记录每个句子所构成的样本数
sample_num_memory = []

""" 写入所有样本 """
f = open('data/training_data/dailydial_pairs.txt', "w+", encoding='utf-8')
# 遍历所有句子的样本列表
for tup in tuples:
    # 记录当前句子的样本数量
    sample_num_memory.append(len(tup))
    # 遍历当前句子的所有样本
    for pir in tup:
        sent1 = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', pir[0])
        sent2 = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', pir[1])
        f.write(sent1+'\t\t'+sent2)
        f.write('\n')
f.close()

""" 写入所有句子的样本数 """
f = open('data/training_data/dailydial_sample_num.txt', "w+")
for sample_size in sample_num_memory:
    f.write(str(sample_size))
    f.write('\n')
f.close()





