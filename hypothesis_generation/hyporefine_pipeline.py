import argparse
import logging
import re
import time
import pickle
import sys

import os
import math
import json

import random
from typing import Callable, Tuple, Union
import torch
import numpy as np

from hypogenic.extract_label import extract_label_register, likert_extract_label

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import LocalVllmWrapper, LLMWrapper, GPTWrapper
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)

from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.algorithm.inference import (
    DefaultInference,
    OneStepAdaptiveInference,
    FilterAndWeightInference,
    TwoStepAdaptiveInference,
    UpperboundInference,
)
from hypogenic.algorithm.replace import DefaultReplace
from hypogenic.algorithm.update import SamplingUpdate, DefaultUpdate
from hypogenic.logger_config import LoggerConfig
from hypogenic.utils import get_results

from hypothesis_agent.data_analysis_agent.generation import TestGeneration
from hypothesis_agent.data_analysis_agent.update import TestUpdate
from hypothesis_agent.literature_review_agent.literature_review import LiteratureAgent
from hypothesis_agent.literature_review_agent.literature_processor.extract_info import BaseExtractor, WholeExtractor
from hypothesis_agent.literature_review_agent.literature_processor.summarize import LLMSummarize
from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
from hypothesis_agent.data_analysis_agent.union_generation import union_hypogenic_and_paper
from hypothesis_agent.data_analysis_agent.inference import MultiHypDefaultInference

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hypoeval_replicate.utils import get_aspect_definition

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("HypoRefine")

def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main(
    model_name = "gpt-4o-mini",
    task_name = "summeval",
):
    use_refine = True # set to True if using HypoRefine, or False for HypoGeniC

    aspects = ["coherence"] # specify the aspects to generate hypotheses

    if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        model_path = "/net/projects/chai-lab/shared_models/Llama-3.3-70B-Instruct"
    elif model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model_path = "/net/projects/chai-lab/shared_models/Meta-Llama-3.1-8B-Instruct"

    if "gpt" in model_name:
        api = GPTWrapper(model_name)
    else:
        api = LocalVllmWrapper(model_name, model_path, gpu_memory_utilization=0.95)

    for aspect in aspects:
        task_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/{task_name}/config.yaml"))
        papers_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../literature/{task_name}/processed"))

        max_num_hypotheses = 20
        max_refine = 6
        num_init = 5
        num_train = 30
        num_test = 30
        num_val = 30
        k = 10
        alpha = 5e-1
        update_batch_size = 5
        num_hypotheses_to_update = 1
        save_every_10_examples = 4
        init_batch_size = 5
        init_hypotheses_per_batch = 5
        cache_seed = None
        temperature = 1e-5
        max_tokens = 4000
        max_concurrent = 64
        seeds = [42]
        prioritize = "balanced" # merging strategy for Union methods
        n_paper_specificity_boost = 0 # deprecated, set to 0

        task = BaseTask(
            task_config_path, extract_label=likert_extract_label
        )

        for seed in seeds:
            definition = get_aspect_definition(task_name, aspect)
            set_seed(seed)
            train_data, test_data, val_data = task.get_data(aspect, definition, num_train, num_test, num_val, seed)
            prompt_class = TestPrompt(task, aspect, definition)
            extractor = WholeExtractor()
            summarizer = LLMSummarize(extractor, api, prompt_class)
            literature_agent = LiteratureAgent(api, prompt_class, summarizer)
            generate_kwargs = {}
            literature_agent.summarize_papers(
                data_file=papers_dir_path,
                cache_seed=cache_seed,
                **generate_kwargs,
            )

            union_hyp_bank = union_hypogenic_and_paper(
                aspect=aspect,
                definition=definition,
                task=task,
                prompt_class=prompt_class,
                literature_agent=literature_agent,
                extractor=extractor,
                api=api,
                train_data=train_data,
                config_path=task_config_path,
                prioritize=prioritize,
                model_name=model_name,
                papers_dir_path=papers_dir_path,
                task_name=task_name,
                custom_dump_path=None,
                max_num_hypotheses=max_num_hypotheses,
                use_refine=use_refine,
                n_paper_specificity_boost=n_paper_specificity_boost,
                num_init=num_init,
                seed=seed,
                k=k,
                alpha=alpha,
                update_batch_size=update_batch_size,
                num_hypotheses_to_update=num_hypotheses_to_update,
                save_every_10_examples=save_every_10_examples,
                init_batch_size=init_batch_size,
                init_hypotheses_per_batch=init_hypotheses_per_batch,
                max_refine=max_refine,
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                temperature=temperature,
                max_tokens=max_tokens,
                **generate_kwargs,
            )
        print("==========================")
        print(union_hyp_bank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--task_name", type=str, default="summeval")
    args = parser.parse_args()
    main(args.model_name, args.task_name)
