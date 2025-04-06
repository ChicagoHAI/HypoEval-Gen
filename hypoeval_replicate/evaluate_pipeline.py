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
from extract_responses import extract_likert_score, batched_extract_likert_scores
from utils import get_aspect_definition, retrieve_hypotheses
from evaluate import batched_get_checklist_reward
from utils import load_summeval_all, load_summeval_test, load_summeval_train
from utils import load_newsroom_train, load_newsroom_test, load_hanna_train, load_hanna_test, load_writingprompt_train, load_writingprompt_test

def main(
    model_name = "gpt-4o-mini",
    task_name = "summeval",
):
    logger = LoggerConfig.get_logger("Evaluate Pipeline")
    aspects = ["coherence"]
    seeds = [42]
    num_select_hyps = 5

    if task_name == "summeval":
        n_test_batch = 40
    elif task_name == "newsroom":
        n_test_batch = 30
    elif task_name == "hanna":
        n_test_batch = 60
    elif task_name == "writingprompt":
        n_test = 300
    else:
        logger.warning(f"Need to specify n_test_batch for task {task_name}")
        n_test_batch = 1

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
    
    prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/{task_name}/config.yaml"))
    with open(prompt_path, 'r') as file:
        prompt_templates = yaml.safe_load(file)
    threshold = 0.5
    max_concurrent = 64
    max_tokens = 4000
    temperature = 1e-5
    generate_kwargs = {"temperature": temperature}

    spearman_corr_list = {}
    pearson_corr_list = {}
    all_results = {}
    if task_name == "writingprompt":
        for aspect in aspects:
            all_results[aspect] = {
                "dataset_level_spearman": [],
                "dataset_level_pearson": [], 
            }
    else:
        for aspect in aspects:
            all_results[aspect] = {
                "summary/story_level_spearman": [],
                "summary/story_level_pearson": [], 
            }

    for aspect in aspects:
        
        # specify the location of training data and generated hypotheses (before selection), default paths provided below
        train_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/{task_name}/train_continuous_{aspect}.json"))
        hyp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypothesis_generation/hypotheses/hypotheses_{task_name}_{aspect}/union/union/{task_name}/{model_name}/hyp_20/refine_6/union_prioritize_balanced_refine_6.json"))
        hyp_bank = retrieve_hypotheses(hyp_path)
        logger.info(f"Size of hypothesis bank before selection: {len(hyp_bank)}")

        for seed in seeds:
            random.seed(seed)
            spearman_corr_list = {}
            pearson_corr_list = {}
            spearman_corr_list[aspect] = []
            pearson_corr_list[aspect] = []
            definition = get_aspect_definition(task_name, aspect)

            if task_name == "summeval":
                train_data = load_summeval_train(train_data_path, seed)
                test_data = load_summeval_test(train_data_path, n_test_batch, seed)
            elif task_name == "newsroom":
                train_data = load_newsroom_train(train_data_path, seed)
                test_data = load_newsroom_test(train_data_path, n_test_batch, seed)
            elif task_name == "hanna":
                train_data = load_hanna_train(train_data_path, seed)
                test_data = load_hanna_test(train_data_path, n_test_batch, seed)
            elif task_name == "writingprompt":
                train_data = load_writingprompt_train(train_data_path, seed)
                test_data = load_writingprompt_test(train_data_path, n_test, seed)
            else:
                logger.warning(f"Need to specify data loading method for {task_name}")
                return

            logger.info(f"Length of training data: {len(train_data)}")
            logger.info(f"Length of test data (n_batch): {len(test_data)}")

            selector_metric = "pearson"
            hyp_reward, _, _ = batched_get_checklist_reward(
                task_name,
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
            # for hyp in sorted_init_hyp_bank.keys():
            #     print(f"Stats for hypothesis: {hyp}")
            #     print(f"{selector_metric} = {hyp_reward[hyp]}")

            selected_hyp_bank = sorted_init_hyp_list[:num_select_hyps]
            
            if task_name == "writingprompt":
                hyp_reward, llm_scores, human_scores = batched_get_checklist_reward(
                    task_name,
                    api,
                    "spearman",
                    prompt_templates,
                    aspect,
                    definition,
                    selected_hyp_bank,
                    test_data,
                    threshold,
                    max_concurrent,
                    max_tokens,
                    **generate_kwargs
                )
                spearman_corr, _ = spearmanr(llm_scores, human_scores)
                pearson_corr, _ = pearsonr(llm_scores, human_scores)
                if math.isnan(spearman_corr):
                    spearman_corr = 1 if all(element==human_scores[0] for element in human_scores) else 0
                if math.isnan(pearson_corr):
                    pearson_corr = 1 if all(element==human_scores[0] for element in human_scores) else 0
                all_results[aspect]["dataset_level_spearman"].append(spearman_corr)
                all_results[aspect]["dataset_level_pearson"].append(pearson_corr)
            else:
                for i in range(0, len(test_data)):
                    print(f"Running on passage {i+1}/{len(test_data)}")
                    hyp_reward, llm_scores, human_scores = batched_get_checklist_reward(
                        task_name,
                        api,
                        "spearman",
                        prompt_templates,
                        aspect,
                        definition,
                        selected_hyp_bank,
                        test_data[i],
                        threshold,
                        max_concurrent,
                        max_tokens,
                        **generate_kwargs
                    )
                    spearman_corr, _ = spearmanr(llm_scores, human_scores)
                    pearson_corr, _ = pearsonr(llm_scores, human_scores)
                    if math.isnan(spearman_corr):
                        spearman_corr = 1 if all(element==human_scores[0] for element in human_scores) else 0
                    if math.isnan(pearson_corr):
                        pearson_corr = 1 if all(element==human_scores[0] for element in human_scores) else 0
                    spearman_corr_list[aspect].append(spearman_corr)
                    pearson_corr_list[aspect].append(pearson_corr)

                all_results[aspect]["summary/story_level_spearman"].append(float(sum(spearman_corr_list[aspect]) / len(spearman_corr_list[aspect])))
                all_results[aspect]["summary/story_level_pearson"].append(float(sum(pearson_corr_list[aspect]) / len(pearson_corr_list[aspect])))

    for aspect in aspects:
        print(f"For task = {task_name}, aspect = {aspect}, seeds = {seeds}")
        for key in all_results[aspect].keys():
            print(f"{key}: {all_results[aspect][key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--task_name", type=str, default="summeval")
    args = parser.parse_args()
    main(
        args.model_name,
        args.task_name,
    )