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

## N-gram evaluation
| Model |\# lag Conv. (Q-R) |  BLEU  | ROUGE1 | ROUGE2 | ROUGEL | d-ROUGE1 | d-ROUGE2 | d-ROUGEL | 
|:---------------|:---------|--------|--------|--------|--------|----------|----------|----------|
| Raw utterance  | None     | 0.3741 | 0.6990 | 0.5318 | 0.6973 | 0.0000   | 0.0000   | 0.0000   |
|----------------|----------|--------|--------|--------|--------|----------|----------|----------|
| T5-NTR         | 3-3      | 0.5421 | 0.8000 | 0.6844 | 0.7778 | 0.2446   | 0.1280   | 0.2382   |
| T5-NTR         | 6-3      | 0.5418 | 0.7996 | 0.6843 | 0.7774 | 0.2434   | 0.1271   | 0.2369   |
|----------------|----------|--------|--------|--------|--------|----------|----------|----------|
| Baseline       | 3-3      | 0.6119 | 0.8316 | 0.7296 | 0.8189 | 0.2574   | 0.1399   | 0.2538   |
| Baseline       | 6-3      | 0.6127 | 0.8320 | 0.7304 | 0.8195 | 0.2579   | 0.1401   | 0.2544   |
|----------------|----------|--------|--------|--------|--------|----------|----------|----------|
| Function-flat  | 3-3      | 0.6246 | 0.8390 | 0.7417 | 0.8265 | 0.2606   | 0.1433   | 0.2572   |
| Function-flat  | 6-6      | 0.6216 | 0.8364 | 0.7391 | 0.8238 | 0.2608   | 0.1428   | 0.2575   |
|----------------|----------|--------|--------|--------|--------|----------|----------|----------|

## Retreival evalauation
| Model |\# lag Conv. (Q-R) |  BLEU  | ROUGE1 | ROUGE2 | 
|:---------------|:---------|--------|--------|--------|
| Raw utterance  | None     | 0.3741 | 0.6990 | 0.5318 |
|----------------|----------|--------|--------|--------|
| T5-NTR         | 3-3      | 
| T5-NTR         | 6-3      | 
|----------------|----------|--------|--------|--------|
| Baseline       | 3-3      | 0.2770 | 0.4467 | 0.7440 |
| Baseline       | 6-3      |
|----------------|----------|--------|--------|--------|
| Function-flat  | 3-3      | 
| Function-flat  | 6-6      | 
| Baseline       | -        | 0.3140 | 0.5018 | 0.8161 |
| Manual         | -        | 0.3889 | 0.6116 | 0.9631 |
|----------------|----------|--------|--------|--------|
