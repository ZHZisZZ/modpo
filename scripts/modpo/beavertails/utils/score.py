from dataclasses import dataclass, field
from typing import Optional
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

from src.utils import (
    disable_progress_bar_non_local_main
)
from scripts.modpo.beavertails.utils.score_model import LlamaForScore

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:

    input_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)

    reward = LlamaForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward', torch_dtype=torch.bfloat16, device_map='auto')
    cost = LlamaForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-cost', torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward')

    generation = load_dataset(script_args.input_dir, split="train")

    results = []
    with torch.no_grad():
        for prompt_response in tqdm.tqdm(generation['prompt_response']):
            input = tokenizer(prompt_response, return_tensors="pt")
            reward_output = reward(input["input_ids"].cuda(), input["attention_mask"].cuda())
            cost_output = cost(input["input_ids"].cuda(), input["attention_mask"].cuda())
            results.append({
                "prompt_response": prompt_response,
                "reward": reward_output.end_scores.item(),
                "cost": cost_output.end_scores.item(),
            })
    
    # raw
    dataset = Dataset.from_list(results)
    dataset.to_json(os.path.join(script_args.output_dir, "raw.jsonl"))

    # mean
    rewards = [result["reward"] for result in results]
    costs = [result["cost"] for result in results]
    mean_reward = sum(rewards) / len(rewards)
    mean_cost = sum(costs) / len(costs)
    with open(os.path.join(script_args.output_dir, "mean.csv"), "w") as f:
        f.write("mean reward, mean cost\n")
        f.write(f"{mean_reward}, {mean_cost}\n")
