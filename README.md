# HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation
This repository provides the code implementation of [HypoEval: Hypothesis-Guided Evaluation of Natural Language Generation](https://arxiv.org/abs/2504.07174), as well as zero-shot evaluators for summarization and story generation using training data from the datasets used in the paper. With only 30 human annotations for each evaluated aspect, **HypoEval** first generates hypotheses that are decomposed dimensions of the evaluated aspect, and then uses a checklist-like approach to combine LLM's Likert scores on each hypothesis to acquire an overall score for an evaluated text. **HypoEval** provides automated interpretable evaluation of natural language generation, with high alignment with human evaluations.

The hypothesis generation module of **HypoEval** is built upon [**HypoRefine**](https://arxiv.org/abs/2410.17309) and [**HypoGeniC**](https://arxiv.org/abs/2404.04326), check out the repository at [**ChicagoHAI/hypothesis-generation**](https://github.com/ChicagoHAI/hypothesis-generation).

![hypoeval_fig1.png](https://github.com/ChicagoHAI/HypoEval-Gen/blob/main/hypoeval_fig1.png?raw=true)

## Use 0-shot Evaluators for Summarization and Story Generation

We provide 0-shot hypothesis-guided evaluators for summaries and story generations, with the hypotheses generated and selected using training data in `data/`.

To use the evaluator for summaries on aspect in `["coherence", "consistency", "informativeness", "fluency", "relevance"]`:

```bash
from hypoeval.evaluator import SummaryEvaluator

evaluator = SummaryEvaluator(model_name=MODEL_NAME, model_path=MODEL_PATH) # (optional) specify model path for local models
evaluated_aspect = "coherence"
summary_list = ["...", "..."]
source_text_list = ["...", "..."]
evaluation_scores = evaluator.batched_evaluate(aspect=evaluated_aspect, summaries=summary_list, source_texts=source_text_list)
```

To use the evaluator for stories on aspect in `["coherence", "cohesiveness", "complexity", "empathy", "engagement", "grammaticality", "likability", "relevance", "surprise"]`:

```bash
from hypoeval.evaluator import StoryEvaluator

evaluator = StoryEvaluator(model_name=MODEL_NAME, model_path=MODEL_PATH) # (optional) specify model path for local models
evaluated_aspect = "coherence"
story_list = ["...", "..."]
story_prompt_list = ["...", "..."]
evaluation_scores = evaluator.batched_evaluate(aspect=evaluated_aspect, stories=story_list, story_prompts=story_prompt_list)
```

## Add new evaluated aspects for summmarization and story generation

Adding a new evaluated aspect requires a small-scale corpus of human evaluation scores on that aspect. Follow the steps below:

1. Preprocess the human evaluation scores similar to `data/summeval/train_continuous_coherence.json` or `data/hanna/train_continuous_coherence.json`:

```bash
# for summarization
new_human_data = {"candidate_summary": summary_list, "source_text": source_text_list, "label": human_score_list}

# for story generation
new_human_data = {"story": story_list, "prompt": story_prompt_list, "label": human_score_list}

with open(f"./data/{TASK_NAME}/train_continuous_{NEW_ASPECT}.json", 'w') as file:
    json.dump(new_human_data, file)
```

2. Modify `get_aspect_definition` in `hypoeval_reproduce/utils.py` to add the definition of new aspect.

3. Generating hypotheses. Modify `hypothesis_generation/hyporefine_pipeline.py` to specify the new aspect, and then run

```bash
python hyporefine_pipeline.py --model_name MODEL_NAME --task_name TASK_NAME
```

4. Hypothesis selection. Modify `hypoeval/summary_evaluate_selection.py` or `hypoeval/story_evaluate_selection.py` to specify the new aspect, then run

```bash
python summary_evaluate_selection.py --model_name MODEL_NAME
```

or

```bash
python story_evaluate_selection.py --model_name MODEL_NAME
```

5. Evaluation. Follow the same steps as [Use 0-shot Evaluators for Summarization and Story Generation](#use-0-shot-evaluators-for-summarization-and-story-generation).

## Reproduce Results

We include all original data for the four datasets in `data_original/` and the training data together with prompts in `data/`.

To reproduce results in the paper:

1. Generating hypotheses. Modify `hypothesis_generation/hyporefine_pipeline.py` to specify the evaluated aspects (e.g. coherence for SummEval), and then run

```bash
python hyporefine_pipeline.py --model_name MODEL_NAME --task_name TASK_NAME
```

2. Hypothesis selection and evaluation. Modify `hypoeval_reproduce/evaluate_pipeline.py` to specify the evaluated aspects and random seeds. Then run

```bash
python evaluate_pipeline.py --model_name MODEL_NAME --task_name TASK_NAME
```

where TASK_NAME should be in `["summeval", "newsroom", "hanna", "writingprompt"]`.

## Citation

Please consider citing our work if it contributes to your research:

```
@misc{li2025hypoevalhypothesisguidedevaluationnatural,
      title={HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation}, 
      author={Mingxuan Li and Hanchen Li and Chenhao Tan},
      year={2025},
      eprint={2504.07174},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.07174}, 
}
```