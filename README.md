# MODPO: Multi-Objective Direct Preference Optimization

Code release for [Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization](https://arxiv.org/pdf/2310.03708.pdf).

TL;DR: Compared to [DPO loss](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/dpo_trainer.py#L413), [MODPO loss](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L142) includes [a margin](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L151-L152) to steer language models by multiple objectives.

## Installation

```bash
create -n modpo python=3.10
conda activate modpo
pip install -r requirements.txt
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
```

## Running MODPO

#### Safety alignment

`sh scripts/modpo/beavertails/run.sh` reproduces the safety alignment experiments from the [MODPO paper](https://arxiv.org/pdf/2310.03708.pdf).

#### Summarization with length penalty

`sh scripts/modpo/summarize_w_length_penalty/run.sh` reproduces the simplified Long-form QA experiments from the [MODPO paper](https://arxiv.org/pdf/2310.03708.pdf). MODPO is applied here to balance human preferences with response length in summarizing extensive texts.

## Other examples

This repository also contains other off-the-shelf tuning recipes:

- SFT (Supervised Fine-tuning): [`scripts/examples/sft/run.sh`]()
- RM (Reward Modeling): [`scripts/examples/rm/run.sh`]()
- DPO (Direct Preference Optimization): [`scripts/examples/dpo/run.sh`]()

To implement new alignment algorithms, please add new trainers at [`src/trainer`](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer).


## Customized datasets

For supported datasets, refer to [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19).
To train on your datasets, add them under [`src/data/raw_data`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data) and modify [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19) accordingly. Please see [`src/data/raw_data/shp`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data/shp.py) for an example.

## Reference

```
@misc{zhou2023onepreferencefitsall,
      title={Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization}, 
      author={Zhanhui Zhou and Jie Liu and Chao Yang and Jing Shao and Yu Liu and Xiangyu Yue and Wanli Ouyang and Yu Qiao},
      year={2023},
      eprint={2310.03708},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

