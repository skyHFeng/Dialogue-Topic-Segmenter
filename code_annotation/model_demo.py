import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForNextSentencePrediction, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import argparse
import random
import re

import warnings
warnings.filterwarnings('ignore')


""" ------------------------------定义loss函数------------------------------ """
def MarginRankingLoss(p_scores, n_scores):
    margin = 1
    scores = margin - p_scores + n_scores
    scores = scores.clamp(min=0)

    return scores.mean()


""" ------------------------------加载分词器------------------------------ """
device = 0
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


""" ------------------------------加载数据------------------------------ """
sample_num_memory = []
id_inputs = []

""" 加载所有句子的样本数 """
# for line in open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_sample_num.txt'):
for line in open('data/training_data/dailydial_sample_num.txt'):
    line = line.strip()
    sample_num_memory.append(int(line))

""" 加载所有句子的所有样本 """
# for line in open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_pairs.txt'):
for line in open('data/training_data/dailydial_pairs.txt', encoding='utf-8'):
    line = line.strip().split('\t\t')
    sent1 = line[0]
    sent2 = line[1]
    encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=128, return_tensors='pt')
    encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=128, return_tensors='pt')
    encoded_pair = encoded_sent1[0].tolist() + encoded_sent2[0].tolist()[1:]
    id_inputs.append(torch.Tensor(encoded_pair))

print('Max sentence length: ', max([len(sen) for sen in id_inputs]))

""" 固定样本表征的维度 """
MAX_LEN = 256
print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

""" 构建各样本的mask矩阵 """
attention_masks = []
for sent in id_inputs:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

""" 按句子打包样本 """
# group samples .....
grouped_inputs = []; grouped_masks = []
count = 0
for i in sample_num_memory:
    grouped_inputs.append(id_inputs[count: count + i])
    grouped_masks.append(attention_masks[count: count + i])
    count = count + i
print('The group number is: ' + str(len(grouped_inputs)))

""" 整合所有句子的正/负样本对 """
# generate pos/neg pairs ....
print('start generating pos and neg pairs ... ')
pos_neg_pairs = []; pos_neg_masks = []
for i in range(len(grouped_inputs)):
    # 当前句子只有第二种负样本
    if len(grouped_inputs[i]) == 2:
        # 加入：正样本-负样本2
        pos_neg_pairs.append(grouped_inputs[i])
        pos_neg_masks.append(grouped_masks[i])
    # 当前句子有两种负样本
    else:
        # 加入：正样本-负样本1、正样本-负样本2、负样本1-负样本2
        pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][1]])
        pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][2]])
        pos_neg_pairs.append([grouped_inputs[i][1], grouped_inputs[i][2]])
        pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][1]])
        pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][2]])
        pos_neg_masks.append([grouped_masks[i][1], grouped_masks[i][2]])

print('there are ' + str(len(pos_neg_pairs)) + ' samples been generated...')
fake_labels = [0] * len(pos_neg_pairs)


""" 划分训练集、验证集 """
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(pos_neg_pairs, fake_labels,
                                                                                    random_state=2018, test_size=0.8)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(pos_neg_masks, fake_labels, random_state=2018, test_size=0.8)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

""" 加载训练集、验证集"""
batch_size = 16
# Create the DataLoader for our train set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)                                 # 数据打包
train_sampler = RandomSampler(train_data)                                                           # 随机采样，相当于打乱数据
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)             # 加载数据
# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


""" ------------------------------加载模型------------------------------ """
""" MLP """
coherence_prediction_decoder = []
coherence_prediction_decoder.append(nn.Linear(768, 768))
coherence_prediction_decoder.append(nn.ReLU())
coherence_prediction_decoder.append(nn.Dropout(p=0.1))
coherence_prediction_decoder.append(nn.Linear(768, 2))
coherence_prediction_decoder = nn.Sequential(*coherence_prediction_decoder)
coherence_prediction_decoder.to(device)

""" 主干模型 """
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=False,
                                                      output_hidden_states=True)
model.cuda(device)


""" ------------------------------加载设置------------------------------ """
epochs = 10
# Create the optimize.
optimizer = AdamW(list(model.parameters()) + list(coherence_prediction_decoder.parameters()), lr=2e-5, eps=1e-8)
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

""" 实例化SummaryWriter对象 """
writer = SummaryWriter(log_dir="logs/dailydialog/2023-05-03")


""" ------------------------------训练和验证------------------------------ """
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    stage_loss = 0

    model.train()
    coherence_prediction_decoder.train()

    for step, batch in enumerate(train_dataloader):

        if step % 1000 == 0 and not step == 0:
            print(str(step) + ' steps done....')

        # 获取样本对（正样本-负样本1、正样本-负样本2），shape: [16, 2, 256]
        b_input_ids = batch[0].to(device)
        # 获取相应mask矩阵
        b_input_mask = batch[1].to(device)
        # 梯度置零
        model.zero_grad()
        coherence_prediction_decoder.zero_grad()

        # 分离正样本，前向传播，shape: [16, 256]
        pos_scores = model(b_input_ids[:, 0, :], attention_mask=b_input_mask[:, 0, :])
        # 获取：hidden_states ——> last_hidden_states ——> [CLS] embeddings, shape: [16, 768]
        pos_scores = pos_scores[1][-1][:, 0, :]
        # 得到正样本一致性分数
        pos_scores = coherence_prediction_decoder(pos_scores)

        # 分离负样本，前向传播，shape: [16, 256]
        neg_scores = model(b_input_ids[:, 1, :], attention_mask=b_input_mask[:, 1, :])
        # 获取：hidden_states ——> last_hidden_states ——> [CLS] embeddings, shape: [16, 768]
        neg_scores = neg_scores[1][-1][:, 0, :]
        # 得到负样本一致性分数
        neg_scores = coherence_prediction_decoder(neg_scores)

        # loss = MarginRankingLoss(pos_scores[0][:,0], neg_scores[0][:,0])
        loss = MarginRankingLoss(pos_scores[:, 0], neg_scores[:, 0])
        total_loss += loss.item()
        # 每100个batch，将loss写入tensorboard
        stage_loss += loss.item()
        if step % 100 == 99:
            writer.add_scalar('training loss', stage_loss / 100, epoch_i * len(train_dataloader) + step)
            stage_loss = 0

        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(coherence_prediction_decoder.parameters()), 1.0)
        # 更新参数
        optimizer.step()
        # 将lr写入tensorboard
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch_i * len(train_dataloader) + step)
        # 更新学习率
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print('=========== the loss for epoch ' + str(epoch_i + 1) + ' is: ' + str(avg_train_loss))

    print("")
    print("Running Validation...")

    max_accuracy = 0

    model.eval()
    coherence_prediction_decoder.eval()

    # 记录所有正样本的一致性分数
    all_pos_scores = []
    # 记录所有负样本的一致性分数
    all_neg_scores = []

    for step, batch in enumerate(validation_dataloader):

        if step % 1000 == 0 and not step == 0:
            print(str(step) + ' steps done....')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        with torch.no_grad():
            pos_scores = model(b_input_ids[:, 0, :], attention_mask=b_input_mask[:, 0, :])
            pos_scores = pos_scores[1][-1][:, 0, :]
            pos_scores = coherence_prediction_decoder(pos_scores)
            neg_scores = model(b_input_ids[:, 1, :], attention_mask=b_input_mask[:, 1, :])
            neg_scores = neg_scores[1][-1][:, 0, :]
            neg_scores = coherence_prediction_decoder(neg_scores)

        # all_pos_scores += pos_scores[0][:,0].detach().cpu().numpy().tolist()
        # all_neg_scores += neg_scores[0][:,0].detach().cpu().numpy().tolist()
        all_pos_scores += pos_scores[:, 0].detach().cpu().numpy().tolist()
        all_neg_scores += neg_scores[:, 0].detach().cpu().numpy().tolist()

    labels = []

    # 遍历所有正/负样本对
    for i in range(len(all_pos_scores)):
        # 正样本一致性大于负样本一致性
        if all_pos_scores[i] > all_neg_scores[i]:
            labels.append(1)
        else:
            labels.append(0)

    # 正确预测正样本一致性大的比例
    accuracy = sum(labels) / float(len(all_pos_scores))
    print(accuracy)

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        PATH = 'save/dailydialog/2023-05-03/weights_nspbert_dailydialog'
        torch.save(model.state_dict(), PATH)
        print("已保存：第" + str(epoch_i + 1) + "轮模型参数")

        # model.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')
        # tokenizer.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')
