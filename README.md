# MODPO: Multi-Objective Direct Preference Optimization

This repo includes a reference implementation of MODPO, an algorithm that extends Direct Preference Optimization (DPO) for multiple alignment objectives with minimal overheads, as described in the paper [Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization](https://arxiv.org/pdf/2310.03708.pdf).


## MODPO adapts DPO for multiple objective with two lines of codes

You can find the MODPO loss in [modpo_trainer.py#L132](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L132), and the DPO loss is in [dpo_trainer.py#L415](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/dpo_trainer.py#L415). The main difference is that MODPO includes an extra margin to makes sure that the language model is steered by more than one objective.

## Installation

#### Create virtual env

```bash
create -n modpo python=3.10
conda activate modpo
```

#### Install

```bash
git clone https://github.com/ZHZisZZ/modpo.git
cd modpo
pip install -r requirements.txt
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
```
#### Log in Wandb

```bash
export WANDB_MODE=online
export WANDB_ENTITY={your_id}
export WANDB_KEY={your_key}
wandb login {your_key}
```

We use `cuda-11.8`. You are free to modify the package versions. However, there is no gurantee that modpo always supports the latest version of all the packages included.


## MODPO examples
We provide all-in-one scripts for each MODPO application in `scripts/modpo`👏.

### Safety alignment

`bash scripts/modpo/beavertails/run.sh`

This set of experiments reproduces the safety alignment experiments from [our paper](https://arxiv.org/pdf/2310.03708.pdf). See wandb reports for experiment results [here](https://api.wandb.ai/links/asap-zzhou/qmn8dwhk).

### Summarization with length penalty

`bash scripts/modpo/summarize_w_length_penalty/run.sh`

This sets of experiments is a simpified version of Long-form QA from [our paper](https://arxiv.org/pdf/2310.03708.pdf). We apply MODPO to balance human preference and length penalty for summarization tasks.

## Other examples

This repository currently also supports other methods

- `supervised fine-tuning`: `scripts/examples/sft/run.sh`
- `reward training`: `scripts/examples/rm/run.sh`
- `dpo fine-tuning`: `scripts/examples/dpo/run.sh`

If you want to add your own alignment algorithms, please add a trainer in `src/trainer` and new example in `scripts/examples`.

## Adding new datasets

Please add your dataset class in `src/data/raw_data`. See `src/data/raw_data/shp` for an example.

### Citing MODPO

If you find MODPO useful, you can use the following BibTeX entry:

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

