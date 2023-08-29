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

| Model | BLEU | ROUGE1 | ROUGE2 | ROUGEL | \# CONV. |
|-------|------|--------|--------|--------|------|
| T5-NTR               | 0.5440 | 0.8000 | 0.6828 | 0.7790 | 3 | 
| T5-NTR               | 0.5310 | 0.7893 | 0.6694 | 0.7686 | 8 | 
| Function-base (flat) | 0.6241 | 0.8325 | 0.7339 | 0.8193 | 8 | 
| Function-base (comp) | 0.3857 | 0.6484 | 0.4914 | 0.6453 | 8 | 
|-------|------|--------|--------|--------|------|

### Out-domain evaluation (ikat-train)
The evaluation dataset in ikat-train/ikat-teset, which are the conversational information seeking dataset that envovled personalized statements.

| Model | PTKBs | \# lag Conv. | BLEU | ROUGE1 | ROUGE2 | ROUGEL |
|-------|-------|--------------|------|--------|--------|--------|
| T5-NTR               | None             | 3  | 0.3417 | 0.5951 | 0.4892 | 0.5454 |
| T5-NTR               | None             | 5  | 0.2927 | 0.5246 | 0.4143 | 0.4690 |
| T5-NTR               | None             | 8  | 0.2927 | 0.5214 | 0.4167 | 0.4708 |
|-------|-------|--------------|------|--------|--------|--------|
| T5-NTR               | ALL as conv      | 3  | 0.3525 | 0.5594 | 0.4536 | 0.5174 |
| T5-NTR               | ALL as conv      | 5  | 0.3165 | 0.5248 | 0.4104 | 0.4798 |
| T5-NTR               | ALL as conv      | 8  | 0.3291 | 0.5301 | 0.4213 | 0.4849 |
|-------|-------|--------------|------|--------|--------|--------|
| T5-NTR               | Selected (truth) | 3  | 0.3488 | 0.5356 | 0.4455 | 0.5154 |
| T5-NTR               | Selected (truth) | 5  | 0.2386 | 0.3964 | 0.3068 | 0.3785 |
| T5-NTR               | Selected (truth) | 8  | 0.2386 | 0.3964 | 0.3068 | 0.3785 |
|-------|-------|--------------|------|--------|--------|--------|
| Function-base (flat) | None             | 3  | 0.6359 | 0.7941 | 0.7314 | 0.7881 |
| Function-base (flat) | None             | 5  | 0.6385 | 0.7805 | 0.7158 | 0.7732 |
| Function-base (flat) | None             | 8  | 0.6195 | 0.7720 | 0.7052 | 0.7588 |
| Function-base (flat) | None             | 10 | 0.6177 | 0.7685 | 0.6959 | 0.7548 |
|-------|-------|--------------|------|--------|--------|--------|
