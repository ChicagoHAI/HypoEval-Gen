import numpy as np
import json
import sys
import os
import yaml
import string
import re
from typing import List, Dict, Any
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM_wrapper import GPTWrapper, LocalVllmWrapper
from LLM_wrapper.logger_config import LoggerConfig

def batched_extract_likert_scores(responses: List[str]):
    logger = LoggerConfig.get_logger("extract_label")
    scores = []
    for response in responses:
        if response is None:
            logger.warning(f"Could not extract label from text: {response}")
            scores.append(-1)
            continue

        response = response.lower()
        patterns = [
            r"final score: (\d+)",
            r"answer is: \$\\boxed{(\d+)}\$"
        ]
        flag = False
        for pattern in patterns:
            match = re.findall(pattern, response)
            if len(match) > 0:
                scores.append(match[-1])
                flag = True
                break
        if flag == False:
            logger.warning(f"Could not extract label from text: {response}")
            scores.append(-1)

    return [float(score) for score in scores]

def extract_likert_score(
    responses,
):
    logger = LoggerConfig.get_logger("extract_label")
    scores = []
    for response in responses:
        if response is None:
            logger.warning(f"Could not extract label from text: {response}")
            scores.append(-1)
            continue

        response = response.lower()
        # pattern = r"final score: (\d+)"
        patterns = [
            r"final score: (\d+)",
            r"answer is: \$\\boxed{(\d+)}\$"
        ]

        flag = False
        for pattern in patterns:
            match = re.findall(pattern, response)
            if len(match) > 0:
                scores.append(match[-1])
                flag = True
                break
        if flag == False:
            logger.warning(f"Could not extract label from text: {response}")
            scores.append(-1)
    return scores