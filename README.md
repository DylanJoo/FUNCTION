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


### Evaluation
| Model | BLEU | ROUGE1 | ROUGE2 | ROUGEL | DONE |
|-------|------|--------|--------|--------|------|
| Function-base (flat) | 0.6204 | 0.8317 | 0.7329 | 0.8185 | <ul><li>[x]</li><li> | 
| T5-NTR               | 0.5757 | 0.8115 | 0.6995 | 0.7915 | <ul><li>[x]</li><li> | 
| Function-base (comp  | 0.5678 | 0.7827 | 0.6685 | 0.7805 | <ul><li>[]</li><li> | 
|-------|------|--------|--------|--------|------|

