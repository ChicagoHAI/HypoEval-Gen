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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM_wrapper import GPTWrapper, LocalVllmWrapper
from LLM_wrapper.logger_config import LoggerConfig
from hypoeval_replicate.extract_responses import extract_likert_score, batched_extract_likert_scores
from hypoeval_replicate.utils import get_aspect_definition, get_substitute_dict


class SummaryEvaluator:

    def __init__(
        self,
        model_name,
        model_path=None,
        prompt_path=os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/newsroom/config.yaml"))
    ):
        logger = LoggerConfig.get_logger("Evaluator")

        self.model_name = model_name
        self.model_path = model_path

        with open(prompt_path, 'r') as file:
            self.prompt_templates = yaml.safe_load(file)
        
        if "gpt" in model_name:
            self.api = GPTWrapper(model_name)
        elif model_path is not None:
            self.api = LocalVllmWrapper(model=model_name, path_name=model_path, gpu_memory_utilization=0.95)
        else:
            logger.warning(f"model {model_name} does not have local path or is not supported, may lead to errors")
            self.api = LocalVllmWrapper(model=model_name, gpu_memory_utilization=0.95)

    def batched_evaluate(
        self,
        aspect,
        source_texts: List[str],
        summaries: List[str],
        definition=None,
        max_tokens=4000,
        max_concurrent=64,
        temperature=1e-5,
        **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("Evaluator")

        if len(source_texts) != len(summaries):
            logger.warning(f"number of source texts != number of summaries, double check")
            source_texts = source_texts[:min(len(source_texts), len(summaries))]
            summaries =  summaries[:min(len(source_texts), len(summaries))]

        hyp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypoeval/selected_hypotheses/summary_{aspect}.json"))
        print(hyp_path)
        if not os.path.exists(hyp_path):
            raise ValueError(f"For summarization evaluation, aspect {aspect} not supported (or hyp path error)")
        with open(hyp_path, 'r', encoding="utf-8", errors="replace") as file:
            all_hyps = json.load(file)
        hyp_list = list(all_hyps.keys())

        if definition is None:
            if aspect in ["coherence", "informativeness", "fluency", "relevance"]:
                definition = get_aspect_definition("newsroom", aspect)
            else:
                definition = get_aspect_definition("summeval", aspect)
        
        prompt_template = self.prompt_templates["prompt_templates"]["hypoeval"]
        prompt_inputs = []
        for i in range(0, len(source_texts)):
            for j in range(0, len(hyp_list)):
                hypothesis = hyp_list[j]
                substitute_dict = {"story": source_texts[i], "summary": summaries[i], "hypothesis": hypothesis, "aspect": aspect, "definition": definition}
                substitute_prompt = {key: string.Template(value).substitute(substitute_dict) for key, value in prompt_template.items()}
                prompt = [
                    {"role": "system", "content": substitute_prompt["system"]},
                    {"role": "user", "content": substitute_prompt["user"]}
                ]
                prompt_inputs.append(prompt)
        
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=None,
            max_concurrent=max_concurrent,
            max_tokens=max_tokens,
            temperature=temperature,
            **generate_kwargs,
        )

        all_decomposed_scores = batched_extract_likert_scores(responses)

        hypothesis_scores = {}
        for i in range(0, len(hyp_list)):
            hypothesis_scores[hyp_list[i]] = []

        for i in range(0, len(source_texts)):
            for j in range(0, len(hyp_list)):
                score = all_decomposed_scores[i * len(hyp_list) + j]
                hypothesis_scores[hyp_list[j]].append(score)

        avr_llm_scores = []
        for i in range(0, len(source_texts)):
            all_checklist_score = 0.0
            for j in range(0, len(hyp_list)):
                all_checklist_score += hypothesis_scores[hyp_list[j]][i]
            all_checklist_score /= len(hyp_list)
            all_checklist_score = round(all_checklist_score, 1)
            avr_llm_scores.append(all_checklist_score)
        
        return avr_llm_scores


class StoryEvaluator:

    def __init__(
        self,
        model_name,
        model_path=None,
        prompt_path=os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/hanna/config.yaml"))
    ):
        logger = LoggerConfig.get_logger("Evaluator")

        self.model_name = model_name
        self.model_path = model_path

        with open(prompt_path, 'r') as file:
            self.prompt_templates = yaml.safe_load(file)
        
        if "gpt" in model_name:
            self.api = GPTWrapper(model_name)
        elif model_path is not None:
            self.api = LocalVllmWrapper(model=model_name, path_name=model_path, gpu_memory_utilization=0.95)
        else:
            logger.warning(f"model {model_name} does not have local path or is not supported, may lead to errors")
            self.api = LocalVllmWrapper(model=model_name, gpu_memory_utilization=0.95)
    
    def batched_evaluate(
        self,
        aspect,
        story_prompts: List[str],
        stories: List[str],
        definition=None,
        max_tokens=4000,
        max_concurrent=64,
        temperature=1e-5,
        **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("Evaluator")

        if len(story_prompts) != len(stories):
            logger.warning(f"number of story prompts != number of stories, double check")
            story_prompts = story_prompts[:min(len(story_prompts), len(stories))]
            stories =  stories[:min(len(story_prompts), len(stories))]

        hyp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../hypoeval/selected_hypotheses/story_{aspect}.json"))
        if not os.path.exists(hyp_path):
            raise ValueError(f"For story generation evaluation, aspect {aspect} not supported (or hyp path error)")
        with open(hyp_path, 'r', encoding="utf-8", errors="replace") as file:
            all_hyps = json.load(file)
        hyp_list = list(all_hyps.keys())

        if definition is None:
            if aspect in ["coherence", "complexity", "empathy", "engagement", "relevance", "surprise"]:
                definition = get_aspect_definition("hanna", aspect)
            else:
                definition = get_aspect_definition("writingprompt", aspect)
        
        prompt_template = self.prompt_templates["prompt_templates"]["hypoeval"]
        prompt_inputs = []
        for i in range(0, len(story_prompts)):
            for j in range(0, len(hyp_list)):
                hypothesis = hyp_list[j]
                substitute_dict = {"story": stories[i], "prompt": story_prompts[i], "hypothesis": hypothesis, "aspect": aspect, "definition": definition}
                substitute_prompt = {key: string.Template(value).substitute(substitute_dict) for key, value in prompt_template.items()}
                prompt = [
                    {"role": "system", "content": substitute_prompt["system"]},
                    {"role": "user", "content": substitute_prompt["user"]}
                ]
                prompt_inputs.append(prompt)
        
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=None,
            max_concurrent=max_concurrent,
            max_tokens=max_tokens,
            temperature=temperature,
            **generate_kwargs,
        )

        all_decomposed_scores = batched_extract_likert_scores(responses)

        hypothesis_scores = {}
        for i in range(0, len(hyp_list)):
            hypothesis_scores[hyp_list[i]] = []

        for i in range(0, len(story_prompts)):
            for j in range(0, len(hyp_list)):
                score = all_decomposed_scores[i * len(hyp_list) + j]
                hypothesis_scores[hyp_list[j]].append(score)

        avr_llm_scores = []
        for i in range(0, len(story_prompts)):
            all_checklist_score = 0.0
            for j in range(0, len(hyp_list)):
                all_checklist_score += hypothesis_scores[hyp_list[j]][i]
            all_checklist_score /= len(hyp_list)
            all_checklist_score = round(all_checklist_score, 1)
            avr_llm_scores.append(all_checklist_score)
        
        return avr_llm_scores