from dataclasses import dataclass
from typing import Dict, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor

@dataclass
class SHPRDP(RawDatasetPreprocessor):
    path: Optional[str] = "stanfordnlp/SHP"
    """
    labels: the preference label -- it is 1 if A is preferred to B; 0 if B is preferred to A.
    """
    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train")
        elif split == "validation":
            return load_dataset(self.path, split="validation")
        elif split == "test":
            return load_dataset(self.path, split="test")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        return {
            "raw_prompt": example["history"],
            "prompt":   self.prompt_template.format(raw_prompt=example["history"]),
            "chosen":   example["human_ref_A"] if example["labels"] == 1 else example["human_ref_B"],
            "rejected": example["human_ref_B"] if example["labels"] == 1 else example["human_ref_A"],
        }


if __name__ == '__main__':
    train_dataset      = SHPRDP().get_preference_dataset(split="train")
    validation_dataset = SHPRDP().get_preference_dataset(split="validation")
    test_dataset       = SHPRDP().get_preference_dataset(split="test")

    sft_train_dataset  = SHPRDP().get_sft_dataset(split="train")
