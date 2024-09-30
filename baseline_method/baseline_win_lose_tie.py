import json
# from vllm import LLM, SamplingParams
import random
import sys
import re
import jsonlines
# import difflib
import warnings
import logging
from tqdm import tqdm
import os
import requests

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
llm = LLM(model="/home/gomall/models/Mistral_7B_Instruct_v0-2")
ground_type = sys.argv[1] if len(sys.argv) > 1 else None
query2ground={}

with open('./rank_ground_dataset/'+str(ground_type)+'/1_'+str(ground_type)+'_ground_dataset.json','r') as file:
    datas=json.load(file)
for question,doc in datas.items():
    if question not in query2ground:
        query2ground[question]={}
    for ground_idx,ground_truth in doc.items():
        if 'ground' in ground_idx:
            query2ground[question][ground_idx]=ground_truth

def winlosetie(r):
    with open('./rank_ground_dataset/'+str(ground_type)+'/'+str(r)+'_'+str(ground_type)+'_ground_dataset.json','r') as file:
        datas = json.load(file)
    golden_array={}

    for position in ['a']:
        output=[]
        for question in datas.keys():
            doc_list=datas[question]['doc_list']
            for doc in tqdm(doc_list):
                prediction=doc['passage']
                new_data={}
                new_data[question]={}
                new_data[question]['candidate_answer']=prediction
                new_data[question]['mistral_wlt_result']={}
                for ground_idx,ground_truth in query2ground[question].items():
                    if isinstance(ground_truth, list):
                        for ground in ground_truth:
                            template_prompt = open('./prompt/no_ref_win_lose_tie.txt', encoding='utf-8').read().strip()
                            if position=="a":
                                prompt = template_prompt.replace('#answer_a', prediction).replace("#answer_b", ground).replace("#question",question)
                            if position=="b":
                                prompt = template_prompt.replace('#answer_a', ground).replace("#answer_b", prediction).replace("#question",question)
                            outputs = llm.generate(prompt, sampling_params)
                            answer = outputs[0].outputs[0].text
                            new_data[question]["mistral_wlt_result"][ground_idx]=answer
                    else:
                        template_prompt = open('./prompt/no_ref_win_lose_tie.txt', encoding='utf-8').read().strip()
                        if position=="a":
                            prompt = template_prompt.replace('#answer_a', prediction).replace("#answer_b", ground_truth).replace("#question",question)
                        if position=="b":
                            prompt = template_prompt.replace('#answer_a', ground_truth).replace("#answer_b", prediction).replace("#question",question)

                        outputs = llm.generate(prompt, sampling_params)
                        answer = outputs[0].outputs[0].text # 从输出对象中获取生成的文本
                        new_data[question]["mistral_wlt_result"][ground_idx]=answer
                        
                output.append(new_data)
        with open('./baseline_winlosetie_evaluate_results/'+str(r)+'_'+str(position)+'_'+str(ground_type)+
                  '_mistral_evaluate.json', 'w') as file:
            json.dump(output, file, indent=4)
winlosetie(1)
winlosetie(2)