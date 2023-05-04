import os
import numpy as np
from numpy import random as np_random
# import random
import copy
import itertools
from os import listdir
from os.path import isfile, join
import shutil
from segeval.window.pk import pk
from segeval.window.windowdiff import window_diff as wd
import re
from transformers import BertTokenizer
import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertForNextSentencePrediction
import statistics
from sklearn.metrics import mean_absolute_error, f1_score


""" ------------------------------深度计算函数------------------------------ """
def depth_score_cal(scores):
	output_scores = []

	# 按一致性分数，遍历话语对（k-1个）
	for i in range(len(scores)):
		# 记录话语对i左侧的最大一致性分数
		lflag = scores[i]
		# 记录话语对i右侧的最大一致性分数
		rflag = scores[i]

		# 第一个话语对
		if i == 0:
			for r in range(i+1, len(scores)):
				if rflag <= scores[r]:
					rflag = scores[r]
				else:
					break
		# 最后一个话语对
		elif i == len(scores):
			for l in range(i-1, -1, -1):              # 倒序遍历
				if lflag <= scores[l]:
					lflag = scores[l]
				else:
					break
		# 中间话语对
		else:
			for r in range(i+1, len(scores)):
				if rflag <= scores[r]:
					rflag = scores[r]
				else:
					break
			for l in range(i-1, -1, -1):
				if lflag <= scores[l]:
					lflag = scores[l]
				else:
					break
		# 计算话语对i的深度（越大话题相关性越小）
		depth_score = 0.5 * (lflag + rflag - 2*scores[i])
		output_scores.append(depth_score)

	return output_scores


""" ------------------------------加载训练好的主干模型------------------------------ """
device = 0
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
MODEL_PATH = 'save/dailydialog/2023-05-03/weights_nspbert_dailydialog'
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda(device)
model.eval()


""" ------------------------------加载测试数据------------------------------ """
path_input_docs = 'data/test_data/'
input_files = [f for f in listdir(path_input_docs) if isfile(join(path_input_docs, f))]
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


""" ------------------------------测试------------------------------ """
c = 0                                  # 记录测试文件数
# pick_num = 3
score_wd = 0; score_mae = 0; score_f1 = 0; score_pk = 0
dp_var = []                            # 记录不同测试文件的分割阈值

# 遍历测试文件
for file in input_files:

	# 过滤文件
	if file not in ['.DS_Store', '196']:
	# if file not in ['.DS_Store']:
		print('*********** The current file is : ' + file + '***********')
		text = []                      # 记录文件句子
		id_inputs = []                 # 记录话语对
		depth_scores = []              # 记录话语对的深度值
		seg_r_labels = []              # 记录真实标签
		seg_r = []                     # 记录真实窗口
		tmp = 0

		# 遍历当前文件中的句子，记录真实标签和窗口
		for line in open('data/test_data/'+file):
			# 未遇到分隔符
			if '================' not in line.strip():
				text.append(line.strip())
				seg_r_labels.append(0)
				tmp += 1
			# 遇到分隔符
			else:
				seg_r_labels[-1] = 1
				seg_r.append(tmp)
				tmp = 0
				
		seg_r.append(tmp)

		# 遍历当前文件中的句子，顺序创建话语对
		for i in range(len(text)-1):
			sent1 = text[i]
			sent2 = text[i+1]
			encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=128, return_tensors='pt')
			encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=128, return_tensors='pt')
			encoded_pair = encoded_sent1[0].tolist() + encoded_sent2[0].tolist()[1:]
			id_inputs.append(torch.Tensor(encoded_pair))

		# 固定话语对表征的维度
		MAX_LEN = 256
		id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

		# 构建各话语对的mask矩阵
		attention_masks = []
		for sent in id_inputs:
			att_mask = [int(token_id > 0) for token_id in sent]
			attention_masks.append(att_mask)

		# 获得测试输入
		test_inputs = torch.tensor(id_inputs).to(device)
		test_masks = torch.tensor(attention_masks).to(device)

		# 话语对前向传播
		scores = model(test_inputs, attention_mask=test_masks)
		# 获取：[CLS]输出 ——> 预测为类别0的概率 ——> 激活
		scores = torch.sigmoid(scores[0][:, 0]).detach().cpu().numpy().tolist()

		# 计算话语对的深度
		depth_scores = depth_score_cal(scores)
		# print(depth_scores)

		# boundary_indice = np.argsort(np.array(depth_scores))[-pick_num:]

		# 计算分割阈值
		threshold = sum(depth_scores)/(len(depth_scores))-0.1*statistics.stdev(depth_scores)
		dp_var.append(statistics.stdev(depth_scores))

		# 记录分割点
		boundary_indice = []
		# 记录预测标签
		seg_p_labels = [0]*(len(depth_scores)+1)

		# 遍历话语对的深度值
		for i in range(len(depth_scores)):
			# 若深度值大于阈值，确定为分割点
			if depth_scores[i] > threshold:
				boundary_indice.append(i)
		# 遍历分割点，修改预测标签
		for i in boundary_indice:
			seg_p_labels[i] = 1

		# 记录预测窗口
		tmp = 0; seg_p = []
		# 遍历预测标签，添加预测窗口
		for fake in seg_p_labels:
			if fake == 1:
				tmp += 1
				seg_p.append(tmp)
				tmp = 0
			else:
				tmp += 1
		seg_p.append(tmp)

		# print(depth_scores)
		# print(threshold)
		# print(seg_p)
		# print(seg_r)

		score_pk += pk(seg_p, seg_r)
		score_wd += wd(seg_p, seg_r)
		score_mae += sum(list(map(abs, np.array(seg_r_labels)-np.array(seg_p_labels))))
		score_f1 += f1_score(seg_r_labels, seg_p_labels, labels=[0, 1], average='macro')
		print(c)
		print(seg_r_labels)
		print(seg_p_labels)
		c += 1

print(c)
print('pk: ', score_pk/c)
print('wd: ', score_wd/c)
print('mae: ', score_mae/c)
print('f1: ', score_f1/c)
print('dp variance: ', sum(dp_var)/c)



