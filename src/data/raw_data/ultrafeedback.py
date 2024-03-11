from dataclasses import dataclass
from functools import partial
from typing import Literal, Dict, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor
from src.utils import print_local_main


def ultrafeedback_transform_to_sft(batched_sample, prompt_template):
    new_batched_sample = {
        "raw_prompt": [],
        "prompt": [],
        "response": [],
    }
    for instruction, completions in zip(batched_sample["instruction"], batched_sample["completions"]):
        # breakpoint()
        for completion in completions:
            new_batched_sample["raw_prompt"].append(instruction)
            new_batched_sample["prompt"].append(prompt_template.format(raw_prompt=instruction))
            new_batched_sample["response"].append(completion['response'])
    return new_batched_sample


def ultrafeedback_transform_to_preference(batched_sample):
    def chosen_id(score_0, score_1):
        if score_0 < score_1:
            return 1
        elif score_0 > score_1:
            return 0
        else:
            return -1

    finegrained_dimensions = ("instruction_following", "honesty", "truthfulness", "helpfulness")
    dimensions = finegrained_dimensions + ("overall",)

    new_batched_sample = {
        "prompt": [],
        "response_0": [],
        "response_1": [],
        **{f"{dimension}_chosen_id": [] for dimension in dimensions}
    }
    for instruction, completions in zip(batched_sample["instruction"], batched_sample["completions"]):
        n_responses = len(completions)

        for j in range(n_responses):
            for k in range(j+1, n_responses):
                new_batched_sample["prompt"].append(instruction)
                new_batched_sample["response_0"].append(completions[j]['response'])
                new_batched_sample["response_1"].append(completions[k]['response'])
                new_batched_sample["overall_chosen_id"].append(
                    chosen_id(
                        completions[j]["overall_score"], 
                        completions[k]["overall_score"]
                    )
                )
                for dimension in finegrained_dimensions:
                    new_batched_sample[f"{dimension}_chosen_id"].append(
                        chosen_id(
                            completions[j]["annotations"][dimension]["Rating"], 
                            completions[k]["annotations"][dimension]["Rating"]
                        )
                    )

    return new_batched_sample


@dataclass
class UltraFeedbackRDP(RawDatasetPreprocessor):
    path: Optional[str] = "OpenBMB/UltraFeedback"
    dimension: Optional[Literal["overall", "instruction_following", "honesty", "truthfulness", "helpfulness"]] = None

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            raise NotImplementedError("test split not implemented for UltraFeedbackRDP")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_id = example[f"{self.dimension}_chosen_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_id}"],
            "rejected": example[f"response_{1-chosen_id}"],
        }

    def get_preference_dataset(self, split):
        dataset = self._get_raw_dataset(split)
        if self.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to preference...")
        dataset = dataset.map(
            ultrafeedback_transform_to_preference,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )
        print_local_main("filtering preference...")
        dataset = dataset.filter(lambda x: x[f"{self.dimension}_chosen_id"] != -1)
        print_local_main("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)

    def get_sft_dataset(self, split, **kwargs):
        if self.dimension:
            return super().get_sft_dataset(split, **kwargs)
        dataset = self._get_raw_dataset(split)
        if self.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to sft...")
        return dataset.map(
            partial(ultrafeedback_transform_to_sft, prompt_template=self.prompt_template),
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )


if __name__ == "__main__":
    num_proc = 4
    overall_dataset = UltraFeedbackRDP(dimension="overall", num_proc=num_proc).get_preference_dataset(split="train")
    sft_dataset     = UltraFeedbackRDP(num_proc=num_proc).get_sft_dataset(split="train")
    breakpoint()
