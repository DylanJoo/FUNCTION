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
| T5-NTR           | None             | 1-0  | 
| T5-NTR           | None             | 1-1  | 
| T5-NTR           | None             | 3-0  |
| T5-NTR           | None             | 3-1  |
| T5-NTR           | None             | 5-0  |
| T5-NTR           | None             | 5-1  |
|------------------|------------------|------|--------|--------|--------|--------|
| T5-NTR           | Selected (truth) | 1-0  | 
| T5-NTR           | Selected (truth) | 3-0  | 0.3488 | 0.5356 | 0.4455 | 0.5154 |
| T5-NTR           | Selected (truth) | 3-1  | 0.2386 | 0.3964 | 0.3068 | 0.3785 |
| T5-NTR           | Selected (truth) | 3-2  ||
|------------------|------------------|------|--------|--------|--------|--------|
| Function-flat    | None             | 3    | 0.6359 | 0.7941 | 0.7314 | 0.7881 |
| Function-flat    | None             | 5    | 0.6385 | 0.7805 | 0.7158 | 0.7732 |
| Function-flat    | None             | 8    | 0.6195 | 0.7720 | 0.7052 | 0.7588 |
| Function-flat    | None             | 10   | 0.6177 | 0.7685 | 0.6959 | 0.7548 |
