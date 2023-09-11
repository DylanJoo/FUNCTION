# Long conversational query reformulation via Fusion-in-Decoder
---
The full name is FUsion-iN-ConversaTION (abbreviated as FUNCTION), a long-contextualized conversational query reformulation model. 

The models and pipeline scripts are in this repository, including
* Data preprocessing
* Training
* Generation (prediction)
* Evaluation

Detail can be found in this [arxiv paper](#).

Requirements
```
transformers
datasets
```
---
### Dataset
We use the training/testing datsaet collected by [qrecc repositary](https://github.com/apple/ml-qrecc/tree/main). 
Further detail can be found in their [paper](https://arxiv.org/abs/2010.04898).
Download the qrecc dataset and store in [dataset](src/dataset/).
```
mkdir src/dataset
wget https://github.com/apple/ml-qrecc/raw/main/dataset/qrecc_data.zip -O src/dataset/
unzip src/dataset/qrecc_data.zip
```

### Modeling
- Function-base
The backbone architecture is based on Fusion-in-decoder; please refer to [model variants](src/models/) for detail.

> Fine-tuned checkpoints are uploded on Huggingface.
```
# our fine-tuned
TBD

# init
google/flan-t5-base
DylanJHJ/fidt5-base-nq
```

- Baseline
A popular baseline is T5 fine-tuned on CANARD dataset. Please refer to their [paper](#), [repo](#) and [model checkpoints](castorini/t5-base-canard).


### In-domain evaluation
The evaluation dataset is qrecc-test; it was based on open-domain conversational question answering dataset. 

| Model |\# lag Conv. (Q-R) |  BLEU  | ROUGE1 | ROUGE2 | ROUGEL | 
|:---------------|:---------|--------|--------|--------|--------|
| Raw utterance  | None     | 0.3741 | 0.6990 | 0.5318 | 0.6973 | 
|----------------|----------|--------|--------|--------|--------|
| T5-NTR         | 3-0      | 0.5464 | 0.7945 | 0.6741 | 0.7703 | 
| T5-NTR         | 3-1      | 0.5501 | 0.8072 | 0.6916 | 0.7840 | 
| T5-NTR         | 3-2      | 0.5496 | 0.8069 | 0.6912 | 0.7840 |
| T5-NTR         | 3-3      | 0.5496 | 0.8071 | 0.6912 | 0.7839 |
|----------------|----------|--------|--------|--------|--------|
| T5-NTR         | 5-0      | 0.5481 | 0.7962 | 0.6759 | 0.7719 |
| T5-NTR         | 5-1      | 0.5520 | 0.8078 | 0.6922 | 0.7844 |
| T5-NTR         | 5-2      | 0.5507 | 0.8075 | 0.6922 | 0.7847 |
| T5-NTR         | 5-3      | 0.5500 | 0.8073 | 0.6918 | 0.7844 |
|----------------|----------|--------|--------|--------|--------|
| Function-comp  | 3-3      | 0.5456 | 0.7984 | 0.6769 | 0.7851 |
| Function-comp  | 5-5      | 0.5483 | 0.8004 | 0.6801 | 0.7869 |
| Function-comp  | 7-7      | 0.5487 | 0.8007 | 0.6808 | 0.7871 | 
| Function-comp  | 10-10    | 0.5486 | 0.8004 | 0.6804 | 0.7870 |
|----------------|----------|--------|--------|--------|--------|
| Function-flat  | 3-3      | 0.6206 | 0.8340 | 0.7348 | 0.8217 |
| Function-flat  | 5-5      | 0.6189 | 0.8328 | 0.7339 | 0.8205 |
| Function-flat  | 6-6      | 0.6186 | 0.8324 | 0.7332 | 0.8202 |
| Function-flat  | 7-7      | 0.6184 | 0.8321 | 0.7328 | 0.8197 |
| Function-flat  | 10-10    | 0.6171 | 0.8310 | 0.7313 | 0.8185 |
|----------------|----------|--------|--------|--------|--------|

 
### Out-domain evaluation (ikat-train)
The evaluation dataset in ikat-train/ikat-teset, which are the conversational information seeking dataset that envovled personalized statements.

| Model | PTKBs | \# lag Conv. (Q:R)         | BLEU   | ROUGE1 | ROUGE2 | ROUGEL |
|:-----------------|:-----------------|:-----|--------|--------|--------|--------|
| Raw utterance    | None             | -    | 0.6603 | 0.8162 | 0.7533 | 0.8149 |
|------------------|------------------|------|--------|--------|--------|--------|
| T5-NTR           | None             | 1-0  | 0.2799 | 0.5714 | 0.4718 | 0.5371 | 
| T5-NTR           | None             | 1-1  | 0.3154 | 0.5920 | 0.4850 | 0.5444 |
| T5-NTR           | None             | 3-0  | 0.2725 | 0.5676 | 0.4712 | 0.5347 |
| T5-NTR           | None             | 3-1  | 0.3195 | 0.6018 | 0.4916 | 0.5551 |
| T5-NTR           | None             | 5-0  | 0.2721 | 0.5632 | 0.4639 | 0.5318 |
| T5-NTR           | None             | 5-1  | 0.3216 | 0.6069 | 0.4978 | 0.5595 |
|------------------|------------------|------|--------|--------|--------|--------|
| T5-NTR           | Selected (truth) | 0-0  | 0.2981 | 0.5768 | 0.4773 | 0.5361 |
| T5-NTR           | Selected (truth) | 1-0  | 0.2934 | 0.5676 | 0.4683 | 0.5265 | 
| T5-NTR           | Selected (truth) | 1-1  | 0.3332 | 0.6058 | 0.5022 | 0.5523 |
| T5-NTR           | Selected (truth) | 3-0  | 0.2888 | 0.5732 | 0.4772 | 0.5344 |
| T5-NTR           | Selected (truth) | 3-1  | 0.3305 | 0.6062 | 0.5000 | 0.5525 |
| T5-NTR           | Selected (truth) | 5-0  | 0.2886 | 0.5733 | 0.4743 | 0.5385 |
| T5-NTR           | Selected (truth) | 5-1  | 0.3319 | 0.6109 | 0.5038 | 0.5557 |
|------------------|------------------|------|--------|--------|--------|--------|
| Function-flat    | None             | 3    | 0.6359 | 0.7941 | 0.7314 | 0.7881 |
| Function-flat    | None             | 5    | 0.6385 | 0.7805 | 0.7158 | 0.7732 |
| Function-flat    | None             | 8    | 0.6195 | 0.7720 | 0.7052 | 0.7588 |
| Function-flat    | None             | 10   | 0.6177 | 0.7685 | 0.6959 | 0.7548 |
