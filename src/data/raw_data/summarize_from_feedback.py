from dataclasses import dataclass
from typing import Dict, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor


@dataclass
class SummarizeFromFeedbackRDP(RawDatasetPreprocessor):
    path: Optional[str] = "openai/summarize_from_feedback"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, 'comparisons', split="train")
        elif split == "validation":
            return load_dataset(self.path, 'comparisons', split="validation").train_test_split(test_size=0.5, seed=0)['train']
        elif split == "test":
            return load_dataset(self.path, 'comparisons', split="validation").train_test_split(test_size=0.5, seed=0)['test']
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        return {
            "raw_prompt": example["info"]["post"],
            "prompt":   self.prompt_template.format(raw_prompt=example["info"]["post"]),
            "chosen":   example["summaries"][example["choice"]]["text"],
            "rejected": example["summaries"][1-example["choice"]]["text"],
        }


if __name__ == '__main__':
    train_dataset      = SummarizeFromFeedbackRDP().get_preference_dataset(split="train")
    validation_dataset = SummarizeFromFeedbackRDP().get_preference_dataset(split="validation")
    test_dataset       = SummarizeFromFeedbackRDP().get_preference_dataset(split="test")
