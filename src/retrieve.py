import json
import argparse
from tqdm import tqdm
import numpy as np
from utils import batch_iterator
from datasets import load_dataset

def sparse_retrieve(queries, qids, args):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(k1=args.k1, b=args.b)

    hits = searcher.batch_search(queries, qids, k=args.k, threads=args.threads)

    results = {}
    for qid in qids:
        results[qid] = [(hit.docid, hit.score) for hit in hits[qid]]
        #[NOTE] sort if needed

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    parser.add_argument("--use_manual", action='store_true', default=False)
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--k1",default=0.9, type=float)
    parser.add_argument("--b", default=0.4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--threads", default=1, type=int)
    # search args for dense
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--dense_retrieval", default=False, action='store_true')
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--q-encoder", type=str, default='facebook/contriever')
    args = parser.parse_args()

    data = load_dataset('json', data_files=args.query)['train']
    n = len(data) // args.batch_size + 1
    qid2query = {k: v for k, v in zip(data['generated_question'], data['qid'])}

    # batch search
    results = {}
    for batch_dataset in tqdm(batch_iterator(data, args.batch_size), total=n):
        if args.use_manual:
            batch_results = sparse_retrieve(
                batch_dataset['rewritten_question'], batch_dataset['qid'], args
            )
        else:
            batch_results = sparse_retrieve(
                batch_dataset['generated_question'], batch_dataset['qid'], args
            )
        results.update(batch_results)

    # write
    run_list = []
    with open(args.output, 'w') as fout:
        for qid in results:
        #     run_list.append({
        #         "Conversation_no": int(qid.split('_')[0]),
        #         "Turn_no": int(qid.split('_')[-1])
        #         "Model_rewrite": qidquery[qid], 
        #         "Model_passages": {k, v for (k, v) in results[qid]},
        #         "Model_answer": "<your-answer-for-questionX>"
        #     })
        # json.dump(run_list, fout, indent=2)
            appeared=[]
            for i, (docid, docscore) in enumerate(results[qid]):
                qid_ = qid.replace('trec_', "")
                qid_ = qid_.replace('nq_', "")
                qid_ = qid_.replace('quac_', "")
                if docid not in appeared:
                    fout.write(f"{qid_} Q0 {docid} {i+1} {docscore} pyserini\n")
                    appeared.append(docid)
    print('done')

# def dense_retrieve(queries, args):
#     import torch
#     from pyserini.search import FaissSearcher
#     from contriever import ContrieverQueryEncoder
#     query_encoder = \
#             ContrieverQueryEncoder(args.q_encoder, device=args.device)
#     searcher = FaissSearcher(args.index_dir, args.q_encoder)
#     if torch.cuda.is_available():
#         searcher.query_encoder.model.to(args.device)
#         searcher.query_encoder.device = args.device
#
#     # serp
#     batch_queries = []
#     serp = {}
#     for index, q in enumerate(tqdm(queries, total=len(queries))):
#         batch_queries.append(q)
#         # form a batch
#         if (len(batch_queries) % args.batch_size == 0 ) or \
#                 (index == len(queries) - 1):
#             results = searcher.batch_search(
#                     batch_queries, batch_queries, 
#                     k=args.k, threads=args.threads
#             )
#
#             for q_ in tqdm(batch_queries):
#                 result = [(hit.docid, float(hit.score)) for hit in results[q_]]
#                 serp[q_] = list(map(list, zip(*result)))
#
#             # clear
#             batch_queries.clear()
#             results.clear()
#         else:
#             continue
#     return serp

