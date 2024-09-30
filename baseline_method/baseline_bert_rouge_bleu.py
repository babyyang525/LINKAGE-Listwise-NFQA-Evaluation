from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import random
import sys
import torch.nn.functional as F
import torch
import re
import jsonlines
import difflib
import warnings
import logging
import os
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from rouge import Rouge
rouge = Rouge()
from tqdm import tqdm
def truncate_and_tokenize(text, max_length=512):
    # 使用BERT的分词器进行分词
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text, add_special_tokens=True)))

    # 截断或填充以适应指定的最大长度
    tokens = tokens[:max_length-2]  # 保留 [CLS] 和 [SEP] 标记

    # 添加特殊标记 [CLS] 和 [SEP]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # 将词汇转换为对应的ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 创建PyTorch张量
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)

    return input_ids_tensor

def calculate_bert_score(sentence1, sentence2, model, tokenizer):
    input_ids1 = truncate_and_tokenize(sentence1)
    input_ids2 = truncate_and_tokenize(sentence2)

    # 获取BERT模型的输出
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)

    # 获取最后一层的隐藏状态
    last_hidden_states1 = outputs1.last_hidden_state
    last_hidden_states2 = outputs2.last_hidden_state

    # 取第一个位置（[CLS]标记）的隐藏状态作为句子表示
    sentence_embedding1 = last_hidden_states1[:, 0, :]
    sentence_embedding2 = last_hidden_states2[:, 0, :]

    # 使用余弦相似度计算句子之间的相似度
    similarity = F.cosine_similarity(sentence_embedding1, sentence_embedding2)

    return similarity.item()

def calculate_pearson_coefficient(seq1, seq2):
    # 计算Pearson相关系数
    if len(seq1)<2 or len(seq2)<2:
        return 0
    pearson_coefficient, _ = pearsonr(seq1, seq2)
    return pearson_coefficient

def calculate_spearman_coefficient(seq1, seq2):
    # 计算Spearman系数
    if len(seq1)<2 or len(seq2)<2:
        return 0
    spearman_coefficient, _ = spearmanr(seq1, seq2)
    return spearman_coefficient

model_name = "../bert-base-uncased/"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
ground_type = sys.argv[1] if len(sys.argv) > 1 else None
def count_score(r):
    with open('./rank_ground_dataset/'+str(ground_type)+'/'+str(r)+'_'+str(ground_type)+'_ground_dataset.json','r') as file:
        datas=json.load(file)
    count=0
    bert_array={}
    bleu_array={}
    golden_array={}
    rouge1_array={}
    rouge2_array={}
    rougel_array={}
    for question in tqdm(datas.keys()):
        doc_list=datas[question]['doc_list']
        if 'ground3' in datas[question]:
            if isinstance(datas[question]['ground3'], list):
                if len(datas[question]['ground3'])>1:
                    ground=datas[question]['ground3'][1]
                else:
                    ground=datas[question]['ground3'][0]
            else:
                ground=datas[question]['ground3']
        elif 'ground2' in datas[question]:
            ground=datas[question]['ground2']
            if isinstance(datas[question]['ground2'], list):
                if len(datas[question]['ground2'])>1:
                    ground=datas[question]['ground2'][1]
                else:
                    ground=datas[question]['ground2'][0]
            else:
                ground=datas[question]['ground2']
        elif 'ground1' in datas[question]:
            ground=datas[question]['ground1']
            if isinstance(datas[question]['ground1'], list):
                if len(datas[question]['ground1'])>1:
                    ground=datas[question]['ground1'][1]
                else:
                    ground=datas[question]['ground1'][0]
            else:
                ground=datas[question]['ground1']
        for doc in doc_list:
            prediction=doc['passage']
            if prediction==ground:
                doc_list.remove(doc)
                continue
        for doc in doc_list:
            prediction=doc['passage']
            scores = rouge.get_scores(prediction, ground)
            rouge_l = scores[0]["rouge-l"]["f"]
            rouge_1 = scores[0]["rouge-1"]["f"]
            rouge_2 = scores[0]["rouge-2"]["f"]
            predict_tokens = prediction.split()
            ground_tokens = ground.split()
            bleu_score = sentence_bleu([ground_tokens], predict_tokens)
            bert_score = calculate_bert_score(doc['passage'], ground, model, tokenizer) 
            # print(rouge_l,rouge_1,rouge_2,bleu_score,bert_score)
            if question not in golden_array:
                golden_array[question]=[]
            golden_array[question].append(doc['label'])
            if question not in rouge1_array:
                rouge1_array[question]=[]
            rouge1_array[question].append(rouge_1)
            if question not in rouge2_array:
                rouge2_array[question]=[]
            rouge2_array[question].append(rouge_2)
            if question not in rougel_array:
                rougel_array[question]=[]
            rougel_array[question].append(rouge_l)
            if question not in bert_array:
                bert_array[question]=[]
            bert_array[question].append(bert_score)
            if question not in bleu_array:
                bleu_array[question]=[]
            bleu_array[question].append(bleu_score)
    ######################
    
    kendall_wlt=0
    spearman_wlt=0
    pearson_wlt=0
    count=0
    total_num=len(rouge1_array)
    print(total_num)
    for question in rouge1_array.keys():
        kendall, _ = kendalltau(np.array(golden_array[question]), np.array(rouge1_array[question]))
        spearman_coefficient = calculate_spearman_coefficient(golden_array[question], rouge1_array[question])

        pearson_coefficient = calculate_pearson_coefficient(golden_array[question], rouge1_array[question])
        if np.isnan(kendall):
            total_num-=1
            continue
        else:
            kendall_wlt+=kendall

        if np.isnan(spearman_coefficient):
            total_num-=1
            continue
        else:
            spearman_wlt+=spearman_coefficient

        if np.isnan(pearson_coefficient):
            total_num-=1
            continue
        else:
            pearson_wlt+=pearson_coefficient
        #################
    rouge1_k= kendall_wlt/total_num
    rouge1_s=spearman_wlt/total_num
    rouge1_p=pearson_wlt/total_num
    ##################################################
    kendall_wlt=0
    spearman_wlt=0
    pearson_wlt=0
    count=0
    total_num=len(rouge2_array)
    for question in rouge2_array.keys():
        kendall, _ = kendalltau(np.array(golden_array[question]), np.array(rouge2_array[question]))
        spearman_coefficient = calculate_spearman_coefficient(golden_array[question], rouge2_array[question])

        pearson_coefficient = calculate_pearson_coefficient(golden_array[question], rouge2_array[question])
        if np.isnan(kendall):
            total_num-=1
            continue
        else:
            kendall_wlt+=kendall

        if np.isnan(spearman_coefficient):
            total_num-=1
            continue
        else:
            spearman_wlt+=spearman_coefficient

        if np.isnan(pearson_coefficient):
            total_num-=1
            continue
        else:
            pearson_wlt+=pearson_coefficient
    rouge2_k= kendall_wlt/total_num
    rouge2_s=spearman_wlt/total_num
    rouge2_p=pearson_wlt/total_num
    ##################################################
    kendall_wlt=0
    spearman_wlt=0
    pearson_wlt=0
    count=0
    total_num=len(rougel_array)
    for question in rougel_array.keys():
        kendall, _ = kendalltau(np.array(golden_array[question]), np.array(rougel_array[question]))
        spearman_coefficient = calculate_spearman_coefficient(golden_array[question], rougel_array[question])

        pearson_coefficient = calculate_pearson_coefficient(golden_array[question], rougel_array[question])
        if np.isnan(kendall):
            total_num-=1
            continue
        else:
            kendall_wlt+=kendall

        if np.isnan(spearman_coefficient):
            total_num-=1
            continue
        else:
            spearman_wlt+=spearman_coefficient

        if np.isnan(pearson_coefficient):
            total_num-=1
            continue
        else:
            pearson_wlt+=pearson_coefficient
    rougel_k= kendall_wlt/total_num
    rougel_s=spearman_wlt/total_num
    rougel_p=pearson_wlt/total_num

    
    kendall_wlt=0
    spearman_wlt=0
    pearson_wlt=0
    count=0
    total_num=len(bert_array)
    for question in bert_array.keys():
        kendall, _ = kendalltau(np.array(golden_array[question]), np.array(bert_array[question]))
        spearman_coefficient = calculate_spearman_coefficient(golden_array[question], bert_array[question])

        pearson_coefficient = calculate_pearson_coefficient(golden_array[question], bert_array[question])
        if np.isnan(kendall):
            total_num-=1
            continue
        else:
            kendall_wlt+=kendall

        if np.isnan(spearman_coefficient):
            total_num-=1
            continue
        else:
            spearman_wlt+=spearman_coefficient

        if np.isnan(pearson_coefficient):
            total_num-=1
            continue
        else:
            pearson_wlt+=pearson_coefficient
    bert_k= kendall_wlt/total_num
    bert_s=spearman_wlt/total_num
    bert_p=pearson_wlt/total_num
    
    kendall_wlt=0
    spearman_wlt=0
    pearson_wlt=0
    count=0
    total_num=len(bleu_array)
    for question in bleu_array.keys():
        kendall, _ = kendalltau(np.array(golden_array[question]), np.array(bleu_array[question]))
        spearman_coefficient = calculate_spearman_coefficient(golden_array[question], bleu_array[question])

        pearson_coefficient = calculate_pearson_coefficient(golden_array[question], bleu_array[question])
        if np.isnan(kendall):
            total_num-=1
            continue
        else:
            kendall_wlt+=kendall

        if np.isnan(spearman_coefficient):
            total_num-=1
            continue
        else:
            spearman_wlt+=spearman_coefficient

        if np.isnan(pearson_coefficient):
            total_num-=1
            continue
        else:
            pearson_wlt+=pearson_coefficient
    bleu_k= kendall_wlt/total_num
    bleu_s=spearman_wlt/total_num
    bleu_p=pearson_wlt/total_num
    return rouge1_k,rouge1_s,rouge1_p,rouge2_k,rouge2_s,rouge2_p,rougel_k,rougel_s,rougel_p,bert_k,bert_s,bert_p,bleu_k,bleu_s,bleu_p



kendal_rouge1=0
spearman_rouge1=0
pearson_rouge1=0
kendal_rouge2=0
spearman_rouge2=0
pearson_rouge2=0
kendal_rougel=0
spearman_rougel=0
pearson_rougel=0
kendal_bert=0
spearman_bert=0
pearson_bert=0
kendal_bleu=0
spearman_bleu=0
pearson_bleu=0
rounds=1
for r in range(rounds):
    rouge1_k,rouge1_s,rouge1_p,rouge2_k,rouge2_s,rouge2_p,rougel_k,rougel_s,rougel_p,bert_k,bert_s,bert_p,bleu_k,bleu_s,bleu_p=count_score(r)
    # bert_k,bert_s,bert_p=count_score(r)
    kendal_rouge1+=rouge1_k
    spearman_rouge1+=rouge1_s
    pearson_rouge1+=rouge1_p
    
    kendal_rouge2+=rouge2_k
    spearman_rouge2+=rouge2_s
    pearson_rouge2+=rouge2_p
    
    kendal_rougel+=rougel_k
    spearman_rougel+=rougel_s
    pearson_rougel+=rougel_p
    
    kendal_bert+=bert_k
    spearman_bert+=bert_s
    pearson_bert+=bert_p
    
    kendal_bleu+=bleu_k
    spearman_bleu+=bleu_s
    pearson_bleu+=bleu_p
    
print(ground_type,":")
print('rouge1:','%.4f'%(kendal_rouge1/rounds),'%.4f'%(spearman_rouge1/rounds),'%.4f'%(pearson_rouge1/rounds))
print('rouge2:','%.4f'%(kendal_rouge2/rounds),'%.4f'%(spearman_rouge2/rounds),'%.4f'%(pearson_rouge2/rounds))
print('rougel:','%.4f'%(kendal_rougel/rounds),'%.4f'%(spearman_rougel/rounds),'%.4f'%(pearson_rougel/rounds))
print('bert:','%.4f'%(kendal_bert/rounds),'%.4f'%(spearman_bert/rounds),'%.4f'%(pearson_bert/rounds))
print('bleu:','%.4f'%(kendal_bleu/rounds),'%.4f'%(spearman_bleu/rounds),'%.4f'%(pearson_bleu/rounds))