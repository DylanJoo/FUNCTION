from tqdm import tqdm
import json
import collections
from datasets import load_dataset
CORPUS_PATH='/home/jhju/datasets/qrecc/collection-paragraph/cc-main-2019.jsonl'

def load_corpus(path=CORPUS_PATH, full=True, docid_list=None):
    corpus = {}
    if docid_list is not None:
        docids = {i: i for i in docid_list}

    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            if docid_list is None:
                corpus[str(item['id'])] = item['contents']
                if full is False and len(corpus) > 100:
                    break
            else:
                if item['id'] in docids:
                    corpus[str(item['id'])] = item['contents']
                    docids.pop(str(item['id']))
                if len(docids) == 0:
                    break
    if docid_list:
        print("Corpus size: (total)", len(corpus), '(left)', len(docids))
    else:
        print("Corpus size: (total)", len(corpus))
    return corpus

def load_qrels(path):
    dataset = \
            load_dataset('json', data_files=path, keep_in_memory=True)['train']
    dataset = dataset.map(
            lambda ex: {"id": f"{ex['Conversation_no']}_{ex['Turn_no']}"}
    )

    qrels = []
    for d in dataset:
        for docid in d['Truth_passages']:
            qrels.append((
                d['id'], docid, d['Truth_rewrite'], d['Truth_answer']
            ))
    return qrels

# def load_qrecc(path):
#     dataset = \
#             load_dataset('json', data_files=path, keep_in_memory=True)['train']
#     dataset = dataset.map(
#             lambda ex: {"id": f"{ex['Conversation_no']}_{ex['Turn_no']}"}
#     )
#
#     dataset_dict = {}
#     for d in dataset:
#         dataset_dict[d['id']] = (d['Question'], d['Rewrite'], d['Answer'])
#
#     return dataset_dict
#
