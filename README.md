# MODPO: Multi-Objective Direct Preference Optimization

This repo includes a reference implementation of MODPO, an algorithm that extends Direct Preference Optimization (DPO) for multiple alignment objectives with minimal overheads, as described in the paper [Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization](https://arxiv.org/pdf/2310.03708.pdf).


## MODPO adapts DPO for multiple objectives with two extra lines of codes

The MODPO loss function is shown in [modpo_trainer.py#L142](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L142) while the DPO loss function is shown in [dpo_trainer.py#L413](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/dpo_trainer.py#L413). MODPO differs in that it includes [an extra margin](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer/modpo_trainer.py#L151-L152) to make sure that the language model is steered by more than one objective.

## Installation

#### Create virtual env

```bash
create -n modpo python=3.10
conda activate modpo
```

#### Install dependencies

```bash
git clone https://github.com/ZHZisZZ/modpo.git
cd modpo
pip install -r requirements.txt
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
```

## MODPO examples

### Safety alignment

`bash scripts/modpo/beavertails/run.sh`

This script reproduces the safety alignment experiments from [our paper](https://arxiv.org/pdf/2310.03708.pdf). See wandb reports for experimental results [here](https://api.wandb.ai/links/asap-zzhou/qmn8dwhk).

### Summarization with length penalty

`bash scripts/modpo/summarize_w_length_penalty/run.sh`

This script reproduces the experiments from [Disentangling Length from Quality in Direct Preference Optimization](https://arxiv.org/abs/2403.19159v1), which is a simpified version of Long-form QA experiments from [our paper](https://arxiv.org/pdf/2310.03708.pdf). We apply MODPO to balance human preference and response length in summarizing long text.

## Other examples

This repository also supports some common training pipline:

- `supervised fine-tuning`: `scripts/examples/sft/run.sh`
- `reward training`: `scripts/examples/rm/run.sh`
- `dpo fine-tuning`: `scripts/examples/dpo/run.sh`

If you want to implement your alignment algorithms, please add new trainers under [`src/trainer`](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer).

## Adding customized datasets

[`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19) lists the datasets currently suppported.
If you want to train on your customized datasets, please add new datasets under [`src/data/raw_data`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data) and modify [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19) accordingly. Please see [`src/data/raw_data/shp`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data/shp.py) for an example.

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

