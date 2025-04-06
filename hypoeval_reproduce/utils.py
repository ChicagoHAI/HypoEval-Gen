import numpy as np
import json
import sys
import os
import yaml
import string
import re
import random
import csv
from typing import List, Dict, Any
from datasets import load_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM_wrapper import GPTWrapper, LocalVllmWrapper
from LLM_wrapper.logger_config import LoggerConfig

def get_aspect_definition(dataset_name, aspect):
    if dataset_name == "summeval":
        if aspect == "coherence":
            return "Coherence measures the quality of all sentences collectively, focusing on how well they fit together and sound natural as a whole."
        elif aspect == "consistency":
            return "Consistency measures whether the facts in the summary align with those in the original article, ensuring all facts are accurately reproduced and no untrue information is introduced."
        elif aspect == "fluency":
            return "Fluency measures the quality of individual sentences, evaluating whether they are well-written and grammatically correct."
        elif aspect == "relevance":
            return "Relevance measures how well the summary captures the key points of the article, considering whether all and only the important aspects are included."
        else:
            print(f"Aspect {aspect} for dataset {dataset_name} not supported!")
            return ""
    elif dataset_name == "newsroom":
        if aspect == "coherence":
            return "Do phrases and sentences of the summary fit together and make sense collectively?"
        elif aspect == "fluency":
            return "Are the individual sentences of the summary well-written and grammatical?"
        elif aspect == "informativeness":
            return "How well does the summary capture the key points of the article?"
        elif aspect == "relevance":
            return "Are the details provided by the summary consistent with the details in the article?"
        else:
            print(f"Aspect {aspect} for dataset {dataset_name} not supported!")
            return ""
    elif dataset_name == "hanna":
        if aspect == "coherence":
            return "Coherence measures whether the story makes sense"
        elif aspect == "empathy":
            return "Empathy measures how well you understood the characters' emotions (regardless of whether you agreed with them)"
        elif aspect == "surprise":
            return "Surprise measures how surprising the end of the story was"
        elif aspect == "engagement":
            return "Engagement measures how much you engaged with the story"
        elif aspect == "complexity":
            return "Complexity measures how elaborate the story is"
        elif aspect == "relevance":
            return "Relevance measures how well the story matches its prompt"
        else:
            print(f"Aspect {aspect} for dataset {dataset_name} not supported!")
            return ""
    elif dataset_name == "writingprompt":
        if aspect == "grammaticality":
            return "How grammatically correct is the text of the story fragment?"
        elif aspect == "cohesiveness":
            return "How well do the sentences in the story fragment fit together?"
        elif aspect == "likability":
            return "How enjoyable do you find the story fragment?"
        elif aspect == "relevance":
            return "How relevant is the story fragment to the prompt?"
        else:
            print(f"Aspect {aspect} for dataset {dataset_name} not supported!")
            return ""
    else:
        print(f"dataset {dataset_name} not supported!")
        return ""

def load_summeval_all():
    # load all summeval 100 batches
    annotation_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_original/summeval.json"))
    all_data = []
    with open(annotation_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            all_data.append(record)
    
    ungrouped_test_data = all_data.copy()
    grouped_test_data = []
    grouped_test_data_dict = {}
    for i in range(0, len(ungrouped_test_data)):
        tmp_filepath = ungrouped_test_data[i]["filepath"]
        sourcefile_name = os.path.basename(tmp_filepath)
        sourcefile_path = f"./cnndm/{sourcefile_name}"
        with open(sourcefile_path, 'r', encoding="utf-8", errors="ignore") as file:
            source_text = file.read()
        if source_text not in grouped_test_data_dict.keys():
            grouped_test_data_dict[source_text] = []
        grouped_test_data_dict[source_text].append(ungrouped_test_data[i])
    
    sorted_grouped_test_data_dict = dict(sorted(grouped_test_data_dict.items()))

    for source_text in sorted_grouped_test_data_dict.keys():
        grouped_test_data.append(sorted_grouped_test_data_dict[source_text])
    
    return grouped_test_data

def load_summeval_train(
    train_data_path,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    train_data = []
    for i in range(0, len(raw_train_data["candidate_summary"])):
        dp = {}
        dp["candidate_summary"] = raw_train_data["candidate_summary"][i]
        dp["decoded"] = dp["candidate_summary"]
        dp["source_text"] = raw_train_data["source_text"][i]
        dp["label"] = raw_train_data["label"][i]
        train_data.append(dp)
    return train_data

def load_summeval_test(
    train_data_path,
    n_test_batch,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    all_data = load_summeval_all()
    test_data = []
    candidate_test_data = []
    for i in range(0, len(all_data)):
        tmp_filepath = all_data[i][0]["filepath"]
        sourcefile_name = os.path.basename(tmp_filepath)
        sourcefile_path = f"./cnndm/{sourcefile_name}"
        with open(sourcefile_path, 'r', encoding="utf-8", errors="ignore") as file:
            source_text = file.read()
        if source_text in raw_train_data["source_text"]:
            continue
        candidate_test_data.append(all_data[i])
    test_data = random.sample(candidate_test_data, n_test_batch)
    return test_data

def load_newsroom_all():
    annotation_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_original/newsroom.json"))
    with open(annotation_filepath, 'r', encoding="utf-8", errors="ignore") as file:
        all_data = json.load(file)
    
    ungrouped_test_data = all_data.copy()
    grouped_test_data = []
    grouped_test_data_dict = {}
    for i in range(0, len(ungrouped_test_data)):
        source_text = ungrouped_test_data[i]["source"]
        if source_text not in grouped_test_data_dict.keys():
            grouped_test_data_dict[source_text] = []
        grouped_test_data_dict[source_text].append(ungrouped_test_data[i])
    
    sorted_grouped_test_data_dict = dict(sorted(grouped_test_data_dict.items()))

    for source_text in sorted_grouped_test_data_dict.keys():
        grouped_test_data.append(sorted_grouped_test_data_dict[source_text])
    
    return grouped_test_data

def load_newsroom_train(
    train_data_path,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    train_data = []
    for i in range(0, len(raw_train_data["candidate_summary"])):
        dp = {}
        dp["candidate_summary"] = raw_train_data["candidate_summary"][i]
        dp["decoded"] = dp["candidate_summary"]
        dp["source_text"] = raw_train_data["source_text"][i]
        dp["label"] = raw_train_data["label"][i]
        train_data.append(dp)
    return train_data

def load_newsroom_test(
    train_data_path,
    n_test_batch,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    all_data = load_newsroom_all()
    test_data = []
    candidate_test_data = []
    for i in range(0, len(all_data)):
        source_text = all_data[i][0]["source"]
        if source_text in raw_train_data["source_text"]:
            continue
        candidate_test_data.append(all_data[i])
    test_data = random.sample(candidate_test_data, n_test_batch)
    return test_data

def load_hanna_all():
    dataset_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_original/hanna.csv"))
    tmp_all_data = []
    with open(dataset_filepath, newline='', encoding="utf-8", errors="replace") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tmp_all_data.append(row)
    tmp_all_data = tmp_all_data[1:]
    prompt_grouped_data = {}
    for i in range(0, int(len(tmp_all_data) / 3)):
        dp1 = tmp_all_data[i * 3]
        dp2 = tmp_all_data[i * 3 + 1]
        dp3 = tmp_all_data[i * 3 + 2]
        new_dp = {}
        new_dp["prompt"] = dp1[1]
        new_dp["human_story"] = dp1[2]
        new_dp["story"] = dp1[3]
        new_dp["relevance"] = [int(dp1[5]), int(dp2[5]), int(dp3[5])]
        new_dp["coherence"] = [int(dp1[6]), int(dp2[6]), int(dp3[6])]
        new_dp["empathy"] = [int(dp1[7]), int(dp2[7]), int(dp3[7])]
        new_dp["surprise"] = [int(dp1[8]), int(dp2[8]), int(dp3[8])]
        new_dp["engagement"] = [int(dp1[9]), int(dp2[9]), int(dp3[9])]
        new_dp["complexity"] = [int(dp1[10]), int(dp2[10]), int(dp3[10])]
        if new_dp["prompt"] not in prompt_grouped_data.keys():
            prompt_grouped_data[new_dp["prompt"]] = []
        prompt_grouped_data[new_dp["prompt"]].append(new_dp)
    
    grouped_all_data = []
    for key in prompt_grouped_data.keys():
        grouped_all_data.append(prompt_grouped_data[key])
    return grouped_all_data

def load_hanna_train(
    train_data_path,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    train_data = []
    for i in range(0, len(raw_train_data["story"])):
        dp = {}
        dp["story"] = raw_train_data["story"][i]
        dp["prompt"] = raw_train_data["prompt"][i]
        dp["label"] = raw_train_data["label"][i]
        train_data.append(dp)
    return train_data

def load_hanna_test(
    train_data_path,
    n_test_batch,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    all_data = load_hanna_all()
    test_data = []
    candidate_test_data = []
    for i in range(0, len(all_data)):
        prompt = all_data[i][0]["prompt"]
        if prompt in raw_train_data["prompt"]:
            continue
        candidate_test_data.append(all_data[i])
    test_data = random.sample(candidate_test_data, n_test_batch)
    return test_data

def load_writingprompt_all():
    dataset_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_original/wp_a.json"))
    with open(dataset_filepath, 'r', encoding="utf-8", errors="replace") as file:
        tmp_all_data = json.load(file)
    return tmp_all_data

def load_writingprompt_train(
    train_data_path,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    train_data = []
    for i in range(0, len(raw_train_data["story"])):
        dp = {}
        dp["story"] = raw_train_data["story"][i]
        dp["prompt"] = raw_train_data["prompt"][i]
        dp["label"] = raw_train_data["label"][i]
        train_data.append(dp)
    return train_data

def load_writingprompt_test(
    train_data_path,
    n_test,
    seed,
):
    random.seed(seed)
    with open(train_data_path, 'r', encoding="utf-8", errors="ignore") as file:
        raw_train_data = json.load(file)
    all_data = load_writingprompt_all()

    tmp_test_data = []
    for i in range(0, len(all_data)):
        if all_data[str(i)]["story"] in raw_train_data["story"] and all_data[str(i)]["prompt"] in raw_train_data["prompt"]:
            continue
        tmp_test_data.append(all_data[str(i)])

    test_data = random.sample(tmp_test_data, n_test)
    return test_data

def get_human_score(
    task_name,
    data_point,
    aspect,
):
    logger = LoggerConfig.get_logger("utils")
    if task_name == "summeval":
        label = float((data_point["expert_annotations"][0][aspect] + data_point["expert_annotations"][1][aspect] + data_point["expert_annotations"][2][aspect]) / 3.0)
        label = round(label, 1)
        return label
    elif task_name == "newsroom":
        return data_point["scores"][aspect]
    elif task_name == "hanna":
        human_score = float(sum(data_point[aspect]) / len(data_point[aspect]))
        human_score = round(human_score, 1)
        return human_score
    elif task_name == "writingprompt":
        if aspect == "grammaticality":
            aspect_key = "grammar"
        elif aspect == "cohesiveness":
            aspect_key = "cohesive"
        elif aspect == "likability":
            aspect_key = "enjoy"
        elif aspect == "relevance":
            aspect_key = "relevant"
        else:
            print(f"aspect {aspect} not supported!")
            return
        human_score = float(sum(data_point[aspect_key]) / len(data_point[aspect_key]))
        human_score = round(human_score, 1)
        return human_score
    else:
        logger.warning(f"Task name {task_name} not supported!")
        return None

def get_substitute_dict(
    task_name,
    dp,
    hypothesis,
    aspect,
):
    logger = LoggerConfig.get_logger("utils")

    definition = get_aspect_definition(task_name, aspect)
    if task_name == "summeval":
        if "candidate_summary" in dp.keys():
            summary = dp["candidate_summary"]
        else:
            summary = dp["decoded"]
        if "source_text" not in dp.keys():
            tmp_filepath = dp["filepath"]
            sourcefile_name = os.path.basename(tmp_filepath)
            sourcefile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data_original/cnndm/{sourcefile_name}"))
            with open(sourcefile_path, 'r', encoding="utf-8", errors="ignore") as file:
                source_text = file.read()
        else:
            source_text = dp["source_text"]
        substitute_dict = {"story": source_text, "summary": summary, "hypothesis": hypothesis, "aspect": aspect, "definition": definition}
        return substitute_dict
    elif task_name == "newsroom":
        if "candidate_summary" in dp.keys():
            summary = dp["candidate_summary"]
        else:
            summary = dp["system_output"]
        if "source_text" not in dp.keys():
            source_text = dp["source"]
        else:
            source_text = dp["source_text"]
        substitute_dict = {"story": source_text, "summary": summary, "hypothesis": hypothesis, "aspect": aspect, "definition": definition}
        return substitute_dict
    elif task_name == "hanna":
        story = dp["story"]
        story_prompt = dp["prompt"]
        substitute_dict = {"story": story, "prompt": story_prompt, "hypothesis": hypothesis, "aspect": aspect, "definition": definition}
        return substitute_dict
    elif task_name == "writingprompt":
        story = dp["story"]
        story_prompt = dp["prompt"]
        substitute_dict = {"story": story, "prompt": story_prompt, "hypothesis": hypothesis, "aspect": aspect, "definition": definition}
        return substitute_dict
    else:
        logger.warning(f"Task name {task_name} not supported!")
        return {}

def retrieve_hypotheses(
    hyp_filepath,
):
    logger = LoggerConfig.get_logger("retrieve_hypotheses")
    with open(hyp_filepath, 'r', encoding="utf-8", errors="ignore") as file:
        data = json.load(file)

    retrieved_hyp_bank = []
    for hyp in data.keys():
        retrieved_hyp = ""
        for i in range(99, -1, -1):
            match_prefix = f"hypothesis{i}. "
            if match_prefix in hyp:
                retrieved_hyp = hyp[len(match_prefix):]
        if retrieved_hyp == "":
            logger.warning(f"hypothesis retrieval for {hyp} failed.")
        retrieved_hyp_bank.append(retrieved_hyp)
    
    return retrieved_hyp_bank