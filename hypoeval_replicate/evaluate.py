import tqdm
import asyncio
import httpx
import numpy as np
import json
import sys
import os
import yaml
import string
import re
import math
from typing import List, Dict, Any
from openai import OpenAI
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM_wrapper import GPTWrapper, LocalVllmWrapper
from LLM_wrapper.logger_config import LoggerConfig
from extract_responses import batched_extract_likert_scores
from utils import get_substitute_dict, get_human_score

def batched_get_checklist_reward(
    task_name,
    api,
    reward,
    prompt_templates,
    aspect,
    definition,
    checklists,
    train_data,
    threshold=0.5,
    max_concurrent=32,
    max_tokens=4000,
    **generate_kwargs,
):
    prompt_template = prompt_templates["prompt_templates"]["hypoeval"]
    prompt_inputs = []
    for i in range(0, len(train_data)):
        for j in range(0, len(checklists)):
            hypothesis = checklists[j]
            substitute_dict = get_substitute_dict(task_name, train_data[i], hypothesis, aspect)
            substitute_prompt = {key: string.Template(value).substitute(substitute_dict) for key, value in prompt_template.items()}
            prompt = [
                {"role": "system", "content": substitute_prompt["system"]},
                {"role": "user", "content": substitute_prompt["user"]}
            ]
            prompt_inputs.append(prompt)
    
    responses = api.batched_generate(
        prompt_inputs,
        cache_seed=None,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        **generate_kwargs,
    )

    checklists_scores = {}
    checklists_reward = {}
    for i in range(0, len(checklists)):
        checklists_scores[checklists[i]] = []
    all_decomposed_scores = batched_extract_likert_scores(responses)

    for i in range(0, len(train_data)):
        for j in range(0, len(checklists)):
            score = all_decomposed_scores[i * len(checklists) + j]
            checklists_scores[checklists[j]].append(score)
    
    human_scores = []
    for i in range(0, len(train_data)):
        if "label" not in train_data[i].keys():
            human_score = get_human_score(task_name, train_data[i], aspect)
        else:
            human_score = float(train_data[i]["label"])
            human_score = round(human_score, 1)
        human_scores.append(human_score)
        
    avr_llm_scores = []
    for i in range(0, len(train_data)):
        all_checklist_score = 0.0
        for j in range(0, len(checklists)):
            all_checklist_score += checklists_scores[checklists[j]][i]
        all_checklist_score /= len(checklists)
        all_checklist_score = round(all_checklist_score, 1)
        avr_llm_scores.append(all_checklist_score)

    for i in range(0, len(checklists)):
        checklist = checklists[i]
        llm_scores = checklists_scores[checklist]
        if reward == "spearman":
            checklists_reward[checklist], _ = spearmanr(human_scores, llm_scores)
            if math.isnan(checklists_reward[checklist]):
                checklists_reward[checklist] = 1 if all(element==human_scores[0] for element in human_scores) else 0
        elif reward == "pearson":
            checklists_reward[checklist], _ = pearsonr(human_scores, llm_scores)
            if math.isnan(checklists_reward[checklist]):
                checklists_reward[checklist] = 1 if all(element==human_scores[0] for element in human_scores) else 0
        else:
            checklists_reward[checklist] = -1e8
            print(f"Reward metric {reward} not supported!")

    return checklists_reward, avr_llm_scores, human_scores