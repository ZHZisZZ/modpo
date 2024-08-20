# MODPO: Multi-Objective Direct Preference Optimization

Code release for [Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization](https://arxiv.org/pdf/2310.03708.pdf).

TL;DR: Compared to [DPO loss](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/dpo_trainer.py#L413), [MODPO loss](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L142) includes [a margin](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L151-L152) to steer language models by multiple objectives.

## Installation

```bash
conda create -n modpo python=3.10
conda activate modpo
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# (optional) pip install flash-attn==2.3.2 --no-build-isolation
```

## Running MODPO

This repository includes two MODPO examples:

- Safety alignment ([`scripts/modpo/beavertails`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/modpo/beavertails)): Balances different values such as safety vs. helpfulness.

- Summarization with length penalty ([`scripts/modpo/summarize_w_length_penalty`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/modpo/summarize_w_length_penalty)): Reduces length bias (verbosity) in summarization.

## Other examples

This repository also contains other off-the-shelf tuning recipes:

- SFT (Supervised Fine-tuning): [`scripts/examples/sft/run.sh`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/examples/sft/run.sh)
- RM (Reward Modeling): [`scripts/examples/rm/run.sh`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/examples/rm/run.sh)
- DPO (Direct Preference Optimization): [`scripts/examples/dpo/run.sh`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/examples/dpo/run.sh)

To implement new alignment algorithms, please add new trainers at [`src/trainer`](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer).


## Customized datasets

For supported datasets, refer to [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19).
To train on your datasets, add them under [`src/data/raw_data`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data) and modify [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19) accordingly. Please see [`src/data/raw_data/shp`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data/shp.py) for an example.

## Reference

```
@inproceedings{zhou2024beyond,
  title={Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization},
  author={Zhou, Zhanhui and Liu, Jie and Shao, Jing and Yue, Xiangyu and Yang, Chao and Ouyang, Wanli and Qiao, Yu},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={10586--10613},
  year={2024}
}
```

