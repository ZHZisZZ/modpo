from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from datasets import Dataset, concatenate_datasets

from src.utils import print_local_main

DEFAULT_PROMPT_TEMPLATE = "\n\nHuman:\n{raw_prompt}\n\nAssistant:\n"

@dataclass
class RawDatasetPreprocessor(ABC):
    path: Optional[str] = None
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    sanity_check: Optional[bool] = False
    num_proc: Optional[int] = 4

    @abstractmethod
    def _get_raw_dataset(self, split) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        # return {
        #     "raw_prompt": str, # optional, useful for generation
        #     "prompt":   str,
        #     "chosen":   str,
        #     "rejected": str,
        # }
        raise NotImplementedError

    def get_preference_dataset(self, split) -> Dataset:
        """
        return a dataset of texts with three keys "prompt", "chosen", "rejected", ("raw_prompt", optional)
        """
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)

    def get_sft_dataset(self, split, chosen_only=True):
        """
        return a dataset of texts with two keys "prompt", "response", ("raw_prompt", optional)
        """
        print_local_main("mapping preference to sft...")
        dataset = self.get_preference_dataset(split)
        chosen_only_dataset = dataset.remove_columns("rejected").rename_column("chosen", "response")
        if chosen_only:
            return chosen_only_dataset
        rejected_only_dataset = dataset.remove_columns("chosen").rename_column("rejected", "response")
        return concatenate_datasets([chosen_only_dataset, rejected_only_dataset])
