import json
import random
import sys
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
from tqdm import tqdm
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
ground_type = sys.argv[1] if len(sys.argv) > 1 else None
few_shot = sys.argv[2] if len(sys.argv) > 2 else None
def count_correlation(r):
    with open('./rank_evaluate_results/'+str(ground_type)+'/'+str(r)+'_'+ground_type+'_'+few_shot+'.json', 'r') as file:
        datas = json.load(file)
    count=0
    doc2score={}
    correlation1=2
    correlation2=1
    for data in datas:
        for question in data.keys():
            result=data[question]['mistral_rank_result']
            doc=data[question]['candidate_answer']
            matches = re.findall(r'\[\[(\d+)\]\]', result)
        if matches:
            rank=int(matches[0])
            doc2score[doc]=(-1)*rank
    with open('./rank_ground_dataset/'+str(ground_type)+'/'+str(r)+'_'+str(ground_type)+'_ground_dataset.json','r') as file:
        datas=json.load(file)
    count=0
    score_array={}
    golden_array={}
    for question in tqdm(datas.keys()):
        doc_list=datas[question]['doc_list']
        for doc in doc_list:
            prediction=doc['passage']
            if question not in golden_array:
                golden_array[question]=[]
            if question not in score_array:
                score_array[question]=[]
            golden_array[question].append(doc['label'])
            if prediction not in doc2score:
                doc2score[prediction]=-2
            score_array[question].append(doc2score[prediction])
    ######################
    kendall_wlt=0
    spearman_wlt=0
    pearson_wlt=0
    count=0
    total_num=len(score_array)
    for question in score_array.keys():
        kendall, _ = kendalltau(np.array(golden_array[question]), np.array(score_array[question]))
        spearman_coefficient = calculate_spearman_coefficient(golden_array[question], score_array[question])
        pearson_coefficient = calculate_pearson_coefficient(golden_array[question], score_array[question])
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
    return kendall_wlt/total_num,spearman_wlt/total_num,pearson_wlt/total_num

kendal=0
spearman=0
pearson=0
rounds=3
kendal,spearman,pearson=count_correlation(1)
print('%.4f'%(kendal),'%.4f'%(spearman),'%.4f'%(pearson))
kendal,spearman,pearson=count_correlation(1)
print('%.4f'%(kendal),'%.4f'%(spearman),'%.4f'%(pearson))
kendal,spearman,pearson=count_correlation(2)
print('%.4f'%(kendal),'%.4f'%(spearman),'%.4f'%(pearson))
