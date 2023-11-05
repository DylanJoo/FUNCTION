"""
Here, we adopt a retriver to retrieve top k passages.
Then, apply select the overalpped passages as provenances.

The retrievers are: (1) BM25 (2) contriever.
"""
import json
from utils import load_corpus

# wiki corpus
corpus = load_corpus('/home/jhju/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl')

# pseudo-corpus
fout = open('dataset/wiki_corpus_for_qrecc.jsonl', 'w')
with open('/tmp2/jhju/CQG-for-Interactive-Search/data/qrecc_provenances_bm25.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        q_serp = item['q_serp'][0]
        ref_serp = item['ref_serp'][0]
        serp = [docid for docid in q_serp if docid in ref_serp][:3]

        if len(serp) == 0:
            print(f"# the query {item['id']} has no overlapped passage set")
        else:
            question = item['question']
            answer = item['answer']
            passages = " | ".join([corpus[docid] for docid in serp])

            fout.write(json.dumps({
                "qid": item['id'], "question": question, "answer": answer, "passsage": passages
            }, ensure_ascii=False)+'\n')
