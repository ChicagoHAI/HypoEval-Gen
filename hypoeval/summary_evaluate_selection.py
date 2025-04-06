import numpy as np
import json
import sys
import os
import yaml
import string
import re
import torch
import argparse
import math
from typing import List, Dict, Any
import random
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM_wrapper import GPTWrapper, LocalVllmWrapper
from LLM_wrapper.logger_config import LoggerConfig
from hypoeval_reproduce.extract_responses import extract_likert_score, batched_extract_likert_scores
from hypoeval_reproduce.utils import get_aspect_definition, retrieve_hypotheses
from hypoeval_reproduce.evaluate import batched_get_checklist_reward
from hypoeval_reproduce.utils import load_summeval_test, load_summeval_train
from hypoeval_reproduce.utils import load_newsroom_train, load_newsroom_test

def main(
    model_name = "gpt-4o-mini",
):
    logger = LoggerConfig.get_logger("Evaluate Pipeline")
    aspects = ["coherence", "informativeness", "consistency", "fluency", "relevance"]
    seed = 42
    num_select_hyps = 5

    if "gpt" in model_name:
        api = GPTWrapper(model_name)
    elif model_name == "meta-llama/Llama-3.3-70B-Instruct":
        model_path = "/net/projects/chai-lab/shared_models/Llama-3.3-70B-Instruct"
        api = LocalVllmWrapper(model=model_name,path_name=model_path, gpu_memory_utilization=0.95)
    elif model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":
        model_path = "/net/projects/chai-lab/shared_models/DeepSeek-R1-Distill-Llama-70B-local"
        api = LocalVllmWrapper(model=model_name,path_name=model_path, gpu_memory_utilization=0.95)
    else:
        logger.warning(f"Please add model path for {model_name}")
        return
    
    prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/newsroom/config.yaml"))
    with open(prompt_path, 'r') as file:
        prompt_templates = yaml.safe_load(file)
    
    threshold = 0.5
    max_concurrent = 64
    max_tokens = 4000
    temperature = 1e-5
    generate_kwargs = {"temperature": temperature}

    selected_hyp_bank = {}
    for aspect in aspects:
        selected_hyp_bank[aspect] = {}

    for aspect in aspects:
        # for hypothesis selection, use newsroom training data for coherence, informativeness, fluency, relevance; use summeval for consistency
        if aspect in ["coherence", "informativeness", "fluency", "relevance"]:
            train_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/newsroom/train_continuous_{aspect}.json"))
            train_data = load_newsroom_train(train_data_path, seed)
            train_data_task_name = "newsroom"
        elif aspect == "consistency":
            train_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/summeval/train_continuous_{aspect}.json"))
            train_data = load_summeval_train(train_data_path, seed)
            train_data_task_name = "summeval"
        else:
            logger.warning(f"Aspect {aspect} not supported")
            continue
        
        if aspect in ["coherence", "fluency", "relevance"]:
            hyp_path_summeval = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypothesis_generation/hypotheses/hypotheses_summeval_{aspect}/union/union/summeval/{model_name}/hyp_20/refine_6/union_prioritize_balanced_refine_6.json"))
            hyp_path_newsroom = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypothesis_generation/hypotheses/hypotheses_newsroom_{aspect}/union/union/newsroom/{model_name}/hyp_20/refine_6/union_prioritize_balanced_refine_6.json"))
            hyp_bank_summeval = retrieve_hypotheses(hyp_path_summeval)
            hyp_bank_newsroom = retrieve_hypotheses(hyp_path_newsroom)
            hyp_bank = hyp_bank_summeval + hyp_bank_newsroom
            logger.info(f"Size of hypothesis bank before selection: {len(hyp_bank)}")
        elif aspect == "informativeness":
            hyp_path_newsroom = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypothesis_generation/hypotheses/hypotheses_newsroom_{aspect}/union/union/newsroom/{model_name}/hyp_20/refine_6/union_prioritize_balanced_refine_6.json"))
            hyp_bank_newsroom = retrieve_hypotheses(hyp_path_newsroom)
            hyp_bank = hyp_bank_newsroom
            logger.info(f"Size of hypothesis bank before selection: {len(hyp_bank)}")
        elif aspect == "consistency":
            hyp_path_summeval = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypothesis_generation/hypotheses/hypotheses_summeval_{aspect}/union/union/summeval/{model_name}/hyp_20/refine_6/union_prioritize_balanced_refine_6.json"))
            hyp_bank_summeval = retrieve_hypotheses(hyp_path_summeval)
            hyp_bank = hyp_bank_summeval
            logger.info(f"Size of hypothesis bank before selection: {len(hyp_bank)}")
        else:
            logger.warning(f"Aspect {aspect} not supported")
            continue
            
        random.seed(seed)
        spearman_corr_list = {}
        pearson_corr_list = {}
        spearman_corr_list[aspect] = []
        pearson_corr_list[aspect] = []
        if aspect in ["coherence", "informativeness", "fluency", "relevance"]:
            definition = get_aspect_definition("newsroom", aspect)
        else:
            definition = get_aspect_definition("summeval", aspect)

        logger.info(f"Length of training data: {len(train_data)}")

        selector_metric = "pearson"
        hyp_reward, _, _ = batched_get_checklist_reward(
            train_data_task_name,
            api,
            selector_metric,
            prompt_templates,
            aspect,
            definition,
            hyp_bank,
            train_data,
            threshold,
            max_concurrent,
            max_tokens,
            **generate_kwargs
        )
        sorted_init_hyp_bank = dict(sorted(hyp_reward.items(), key=lambda item: item[1], reverse=True))
        sorted_init_hyp_list = []
        for hyp in sorted_init_hyp_bank.keys():
            sorted_init_hyp_list.append(hyp)

        selected_hyp_list = sorted_init_hyp_list[:num_select_hyps]
        for hyp in selected_hyp_list:
            selected_hyp_bank[aspect][hyp] = hyp_reward[hyp]
    
    for aspect in aspects:
        selected_hyp_path = f"./selected_hypotheses/summary_{aspect}.json"
        with open(selected_hyp_path, 'w') as file:
            json.dump(selected_hyp_bank[aspect], file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    main(
        args.model_name,
    )