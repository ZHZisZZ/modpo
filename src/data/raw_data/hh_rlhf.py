# adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py#L82
from dataclasses import dataclass
from typing import Dict, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor


def preprocess_anthropic_prompt_and_response(prompt_and_response):
    prompt_and_response = prompt_and_response.replace("\n\nHuman: ", "\n\nHuman:\n")
    prompt_and_response = prompt_and_response.replace("\n\nAssistant: ", "\n\nAssistant:\n")
    return prompt_and_response

def extract_anthropic_prompt_and_response(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


@dataclass
class HhRlhfRDP(RawDatasetPreprocessor):
    path: Optional[str] = "Anthropic/hh-rlhf"

    # def __post_init__(self):
    #     assert self.prompt_template == "\n\nHuman: {prompt}\n\nAssistant:"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return load_dataset(self.path, split="test")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["chosen"]   = preprocess_anthropic_prompt_and_response(example["chosen"])
        example["rejected"] = preprocess_anthropic_prompt_and_response(example["rejected"])
        prompt = extract_anthropic_prompt_and_response(example["chosen"])
        return {
            "prompt":   prompt,
            "chosen":   example["chosen"][len(prompt) :],
            "rejected": example["rejected"][len(prompt) :],
        }


if __name__ == "__main__":
    train_dataset      = HhRlhfRDP(num_proc=1).get_preference_dataset(split="train")
    validation_dataset = HhRlhfRDP(num_proc=1).get_preference_dataset(split="validation")
    test_dataset       = HhRlhfRDP(num_proc=1).get_preference_dataset(split="test")
