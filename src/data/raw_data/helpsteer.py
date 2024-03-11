from typing import Dict
from dataclasses import dataclass
from datasets import load_dataset
from typing import Literal, Optional

from .utils import RawDatasetPreprocessor
from src.utils import print_local_main


def helpsteer_transform_to_preference(batched_sample):
    def chosen_id(score_0, score_1):
        if score_0 < score_1:
            return 1
        elif score_0 > score_1:
            return 0
        else:
            return -1

    finegrained_dimensions = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
    dimensions = finegrained_dimensions + ("overall",)

    debatched_sample = [{k:batched_sample[k][i] for k in batched_sample.keys()} for i in range(len(batched_sample["prompt"]))]

    new_batched_sample = {
        "prompt": [],
        "response_0": [],
        "response_1": [],
        **{f"{dimension}_chosen_id": [] for dimension in dimensions}
    }
    mini_debatch = []
    for i, sample in enumerate(debatched_sample):
        mini_debatch.append(sample)
        if i != len(debatched_sample) - 1 and sample["prompt"] == debatched_sample[i+1]["prompt"]:
            continue

        for j in range(len(mini_debatch)):
            for k in range(j+1, len(mini_debatch)):
                new_batched_sample["prompt"].append(mini_debatch[j]["prompt"])
                new_batched_sample["response_0"].append(mini_debatch[j]["response"])
                new_batched_sample["response_1"].append(mini_debatch[k]["response"])
                new_batched_sample["overall_chosen_id"].append(
                    chosen_id(
                        sum(mini_debatch[j][dimension] for dimension in finegrained_dimensions),
                        sum(mini_debatch[k][dimension] for dimension in finegrained_dimensions),
                    )
                )
                for dimension in finegrained_dimensions:
                    new_batched_sample[f"{dimension}_chosen_id"].append(
                        chosen_id(
                            mini_debatch[j][dimension], 
                            mini_debatch[k][dimension],
                        )
                    )

        mini_debatch = []

    return new_batched_sample


@dataclass
class HelpSteerRDP(RawDatasetPreprocessor):
    path: Optional[str] = "nvidia/HelpSteer"
    # None for sft
    dimension: Optional[Literal["overall", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]] = None

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train")
        elif split == "validation":
            return load_dataset(self.path, split="validation")
        elif split == "test":
            raise NotImplementedError("test split not implemented for helpsteer")
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
        assert self.dimension, "preference dimension has to be specified"
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: 
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to preference...")
        dataset = dataset.map(
            helpsteer_transform_to_preference,
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
            lambda sample: {
                "raw_prompt": sample["prompt"],
                "prompt": self.prompt_template.format(raw_prompt=sample["prompt"]), 
                "response": sample["response"], 
            },
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )


if __name__ == "__main__":
    num_proc = 4
    helpful_dataset = HelpSteerRDP(dimension="helpfulness", num_proc=num_proc).get_preference_dataset(split="train")
    overall_dataset = HelpSteerRDP(dimension="overall", num_proc=num_proc).get_preference_dataset(split="train")
    sft_dataset     = HelpSteerRDP(num_proc=num_proc).get_sft_dataset(split="train")
    breakpoint()
