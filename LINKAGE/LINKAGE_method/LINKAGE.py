from transformers import AutoModelForCausalLM, AutoTokenizer
import extract_ground
# from vllm import LLM, SamplingParams
import json
import random
import sys
import re
import jsonlines
# import difflib
import warnings
import logging
import os
from tqdm import tqdm
import requests
def is_folder_empty(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return None

    # 获取文件夹中的所有文件和文件夹列表
    contents = os.listdir(folder_path)

    # 如果文件夹中没有任何内容，则为空
    if not contents:
        return True
    else:
        return False


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
llm = LLM(model="/home/gomall/models/Mistral_7B_Instruct_v0-2")
ground_type = sys.argv[1] if len(sys.argv) > 1 else None
rounds=3
few_shot = sys.argv[2] if len(sys.argv) > 2 else None
query2ground={}

def create_query2ground(r):
    with open('./rank_ground_dataset/'+str(ground_type)+'/'+str(r)+'_'+str(ground_type)+'_ground_dataset.json','r') as file:#3 2 1 0
        datas=json.load(file)
    for question,doc in datas.items():
        if question not in query2ground:
                query2ground[question]={}
        if 'ground3' in doc:
            query2ground[question]['ground3']=doc['ground3']
        if 'ground2' in doc:
            query2ground[question]['ground2']=doc['ground2']
        if 'ground1' in doc:
            query2ground[question]['ground1']=doc['ground1']
        if 'ground0' in doc:
            query2ground[question]['ground0']=doc['ground0']
    return query2ground
def create_ground_reference(query):
    type2score={}
    type2score['ground3']='4'
    type2score['ground2']='3'
    type2score['ground1']='2'
    type2score['ground0']='1'
    ground_str=""
    idx=1
    flag=0
    for ground_type,ground in query2ground[query].items():
        if isinstance(ground, list):
            for g in ground:
                ground_str=ground_str+"Answer"+" "+str(idx)+":"+g+'\n'
                idx+=1
        else:
            ground_str=ground_str+"Answer"+" "+str(idx)+":"+ground+'\n'
            idx+=1
                
    return ground_str
def rank_4_ground(r):
    with open('./rank_ground_dataset/'+str(ground_type)+'/'+str(r)+'_'+str(ground_type)+'_ground_dataset.json','r') as file:#3 2 1 0
        datas = json.load(file)
    golden_array={}
    output=[]
    count = 0
    for question in tqdm(datas.keys()):
        doc_list=datas[question]['doc_list']
        query2ground=create_query2ground(r)
        ground_list=query2ground[question]
        k=0
        for doc in doc_list:
            prediction=doc['passage']
            new_data={}
            new_data[question]={}
            new_data[question]['candidate_answer']=prediction
            new_data[question]['mistral_rank_result']=""
            multi_ground=create_ground_reference(question)
            template_prompt = open('./prompt/'+str(few_shot)+'.txt', encoding='utf-8').read().strip()
            prompt = template_prompt.replace('#ground', multi_ground).replace("#candidate", prediction).replace("#question",question)
            outputs = llm.generate(prompt, sampling_params)
            answer = outputs[0].outputs[0].text # 从输出对象中获取生成的文本
            new_data[question]["mistral_rank_result"]=answer
            output.append(new_data)
    with open('./rank_evaluate_results/'+str(ground_type)+'/'+str(r)+'_'+ground_type+'_'+few_shot+'.json', 'w') as file:
        json.dump(output, file, indent=4)

for r in range(rounds):
    rank_4_ground(r)