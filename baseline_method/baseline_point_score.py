from vllm import LLM, SamplingParams
import json
import random
import sys
import re
import jsonlines
import difflib
import warnings
import logging
from tqdm import tqdm
import os
import requests

device = "cuda"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
llm = LLM(model="/models/Mistral_7B_Instruct_v0-2")
ground_type = sys.argv[1] if len(sys.argv) > 1 else None
need_ref = sys.argv[2] if len(sys.argv) > 2 else None
query2ground={}
with open('./rank_ground_dataset/'+str(ground_type)+'/1_'+str(ground_type)+'_ground_dataset.json','r') as file:
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
def create_ground_reference_with_score(query):
    type2score={}
    type2score['ground3']='4'
    type2score['ground2']='3'
    type2score['ground1']='2'
    type2score['ground0']='1'
    ground_str=""
    idx=1
    for ground_type,ground in query2ground[query].items():
        if isinstance(ground, list):
            for g in ground:
                ground_str=ground_str+"Answer"+" "+str(idx)+":"+g+' Score:'+type2score[ground_type]+'\n'
                idx+=1
        else:
            ground_str=ground_str+"Answer"+" "+str(idx)+":"+ground+' Score:'+type2score[ground_type]+'\n'
            idx+=1
    return ground_str
def create_ground_reference_no_score(query):
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
def point(r):
    with open('./rank_ground_dataset/'+str(ground_type)+'/'+str(r)+
              '_'+str(ground_type)+'_ground_dataset.json','r') as file:
        datas = json.load(file)
    golden_array={}
    output=[]
    print(need_ref,ground_type)
    for question in datas.keys():
        doc_list=datas[question]['doc_list']
        k=0
        for doc in tqdm(doc_list):
            prediction=doc['passage']
            new_data={}
            new_data[question]={}
            new_data[question]['candidate_answer']=prediction
            prompt=""
            template_prompt = open('./prompt/'+str(need_ref)+'.txt', encoding='utf-8').read().strip()
            if 'no_ref' in need_ref:
                prompt = template_prompt.replace('#candidate', prediction).replace("#question",question)
            elif 'with_ref' in need_ref and '10' in need_ref:
                multi_ground=create_ground_reference_no_score(question)
                prompt = template_prompt.replace('#ground', multi_ground).replace("#candidate", prediction).replace("#question",question)
            elif 'with_ref' in need_ref and '4' in need_ref:
                multi_ground=create_ground_reference_with_score(question)
                prompt = template_prompt.replace('#ground', multi_ground).replace("#candidate", prediction).replace("#question",question)
            outputs = llm.generate(prompt, sampling_params)
            answer = outputs[0].outputs[0].text # 从输出对象中获取生成的文本
            new_data[question]["mistral_wlt_result"]=answer
            output.append(new_data)
    with open('./baseline_point_score_evaluate_results/'+str(r)+'_'+str(need_ref)+'_'+str(ground_type)+'.json', 'w') as file:
        json.dump(output, file, indent=4)
    print(need_ref,ground_type)
point(1)
