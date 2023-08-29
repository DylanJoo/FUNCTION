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
| T5-NTR               | ////// | ////// | ////// | ////// | 8 | 
| Function-base (flat) | 0.6241 | 0.8325 | 0.7339 | 0.8193 | 8 | 
| Function-base (comp) | 0.3857 | 0.6484 | 0.4914 | 0.6453 | 8 | 
|-------|------|--------|--------|--------|------|

### Out-domain evaluation (ikat-train)
The evaluation dataset in ikat-train/ikat-teset, which are the conversational information seeking dataset that envovled personalized statements.

| Model | PTKBs | \# lag Conv. | BLEU | ROUGE1 | ROUGE2 | ROUGEL |
|-------|-------|--------------|------|--------|--------|--------|
| T5-NTR               | Selected (truth) | 1  | 0.4326 | 0.6865 | 0.5959 | 0.6574 |
| T5-NTR               | Selected (truth) | 3  | 0.3488 | 0.5356 | 0.4455 | 0.5154 |
| T5-NTR               | Selected (truth) | 5  | 0.2386 | 0.3964 | 0.3068 | 0.3785 |
| T5-NTR               | ALL as conv      | 1  | 0.4125 | 0.6770 | 0.5790 | 0.6444 |
| T5-NTR               | ALL as conv      | 3  | 0.3188 | 0.4966 | 0.3882 | 0.4711 |
| T5-NTR               | ALL as conv      | 5  | 0.2540 | 0.4134 | 0.3150 | 0.3984 |
|-------|------|--------|--------|--------|------|
