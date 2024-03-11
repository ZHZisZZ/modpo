from dataclasses import dataclass
from typing import Dict, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor


@dataclass
class StackExchangePairedRDP(RawDatasetPreprocessor):
    path: Optional[str] = "lvwerra/stack-exchange-paired"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train")
        elif split == "validation":
            return load_dataset(self.path, split="validation").train_test_split(test_size=0.5, seed=0)['train']
        elif split == "test":
            return load_dataset(self.path, split="validation").train_test_split(test_size=0.5, seed=0)['test']
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        return {
            "raw_prompt": example["question"],
            "prompt":   self.prompt_template.format(raw_prompt=example["question"]),
            "chosen":   example["response_j"],
            "rejected": example["response_k"],
        }


if __name__ == '__main__':
    path = "lvwerra/stack-exchange-paired"
    train_dataset      = StackExchangePairedRDP().get_preference_dataset(split="train")
    validation_dataset = StackExchangePairedRDP().get_preference_dataset(split="validation")
    test_dataset       = StackExchangePairedRDP().get_preference_dataset(split="test")
