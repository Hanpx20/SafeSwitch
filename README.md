<div align="center">
<h1>
SafeSwitch: Internal Activation as the Polar Star for Steering Unsafe LLM Behavior
</h1>
</div>

<p align="center">
<a href="111" target="_blank">Paper</a> • <a href="111" target="_blank">Models</a>
</p>




## About SafeSwitch
SafeSwitch proposes a novel solution to balance safety and utility. Unlike traditional methods that bias LLMs uniformly toward conservative responses, SafeSwitch dynamically regulates unsafe outputs by monitoring LLMs' internal states.

![](figures/main_fig.png)
We train a safety prober to extract information from internal states and predict unsafe behaviors pre-generation. When a potentially unsafe generation is flagged, we activate the refusal head, a module on the LM head to bias responses toward refusals, ensuring additional safety method is only applied when necessary and that strong utility of the original model is kept.

## Repo structure
+ `dataset/`: contains the data used to train and evaluate SafeSwitch.

    + `sorry-bench-plus.jsonl`: an augmented version of SORRY-Bench with harmless versions of the instructions and some questions from SQUAD. Used to train and evaluate safety probers.
    + `sorry-bench-train.jsonl`, `trustllm-misuse_train.jsonl`, `trustllm-jailbreak_train.jsonl`: unsafe instructions used to train the refusal head.
    + `judge_prompts.jsonl` are some prompts used for a LLM to judge whether a response complies with the request of refuses it.
    + The rest are evaluation benchmarks. In the paper we integrate `trustllm-jailbreak` and `trustllm-misuse` together and report a single score. `trustllm-exaggerated_safety` corresponds to `Over Refusal` in the paper.
+ `src/data/`: contains code to obtain the datasets.
+ `src/prober/`: contains code to train and evaluate safety probers.
+ `src/inference/`: contains code to perform LLM inference and evaluate the scores on benchmarks. Specially, `head_train.py` is used to train the refusal head.
+ `src/analysis/`: contains code of analytical experiments in the paper. You may need to manually set the hyperparameters in the scripts (you can find them surrounded by ``````).

## Train your SafeSwitch

### Step 0: Prepare the environment

+ `python>=3.10` is required for this repo.
+ It's recommended to use pip package manager. Run `pip install -r requirements.txt` to install all requirements.
+ Run `cd alpace_eval` and `pip install -e .` to install the alpace-eval package.
+ Also, remember to set the system variables according to your environnment before using any of the bash scripts below :)


### Step 1: Train the Prober

Set the `model_list` parameter in `bash_scripts/train_prober_pipeline.sh` and run the script to train and evaluate safety probers.


### Step 2: Train the Refusal Head

Set the `model_list` parameter in `bash_scripts/train_refusal_head.sh` and run the script to train the refusal head. The output directory should contain the whole LM model (where only the LM head is different from the original model) as well as a copy for the refusal head alone.

### Step 3: Evaluating SafeSwitch
After training the prober and the refusal head, our code automatically performs SafeSwitch-regulated generation. You can run the evaluation with: `bash_scripts/eval_pipeline.sh`.

You can also run the following scrpit to interact with Safeswitch:
```
python src/safeswitch_pipeline.py --model Llama-3.1-8B-Instruct \
    --llm_dir /shared/nas2/shared/llms \
    --classifier_dir /shared/nas2/ph16/toxic/outputs/classifier \
    --refusal_head_dir /shared/nas2/ph16/toxic/finetuned_LM_head
```




## Cite this paper
If you find this repo or the paper useful, please cite:
```

```


