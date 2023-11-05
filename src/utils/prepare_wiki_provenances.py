"""
Here, we adopt a retriver to retrieve top k passages.
Then, apply select the overalpped passages as provenances.

The retrievers are: (1) BM25 (2) contriever.
"""
import json
import argparse
from tools import load_corpus
QRECC_PROVENANCES='/tmp2/jhju/CQG-for-Interactive-Search/data/qrecc_provenances_bm25.jsonl'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--wiki_corpus", type=str)
    parser.add_argument("--qrecc_provenance", type=str)
    parser.add_argument("--output_corpus", type=str)
    args = parser.parse_args()

    # wiki corpus
    corpus = load_corpus(args.wiki_corpus)

    # pseudo-corpus
    fout = open(args.output_corpus, 'w')
    with open(args.qrecc_provenance, 'r') as f:
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
