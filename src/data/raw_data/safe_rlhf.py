from dataclasses import dataclass
from abc import ABC
from typing import Dict, Literal, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor

@dataclass
class PKUSafeRlhfRDPBase(RawDatasetPreprocessor, ABC):
    dimension: Literal["safer", "better"] = "better"

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_idx = example[f"{self.dimension}_response_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_idx}"],
            "rejected": example[f"response_{1-chosen_idx}"],
        }

@dataclass
class PKUSafeRlhfRDP(PKUSafeRlhfRDPBase):
    path: Optional[str] = "PKU-Alignment/PKU-SafeRLHF"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return load_dataset(self.path, split="test")
        else:
            raise NotImplementedError


@dataclass
class PKUSafeRlhf10KRDP(PKUSafeRlhfRDPBase):
    path: Optional[str] = "PKU-Alignment/PKU-SafeRLHF-10K"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            raise NotImplementedError("PKU-Alignment/PKU-SafeRLHF-10K is for development, no test set available.")
        else:
            raise NotImplementedError


if __name__ == '__main__':
    safer10k_train_dataset = PKUSafeRlhf10KRDP(dimension="safer").get_preference_dataset(split="train")
    better10k_train_dataset = PKUSafeRlhf10KRDP(dimension="better").get_preference_dataset(split="train")
    breakpoint()
