# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
from tqdm import tqdm
import fire
import json
from llama import Llama
from utils import batch_iterator
from datasets import load_dataset
from utils import load_qrels, load_corpus, normalize
import pickle

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1,
    top_p: float = 0.95,
    max_seq_len: int = 256,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    filename: str = None
):

    # prepare dataset
    ## 1. original setup from scai. Failed to match with qrels)
    # dataset = load_qrels('dataset/scai_qrecc_train_qrel.json')
    # docid_list = [item[1] for item in qrels]
    # corpus = load_corpus('dataset/cc-main-2019.jsonl', full=True)
    ## 2. alternative wiki corpus
    dataset = load_dataset('json', 
            data_files='dataset/wiki_corpus_for_qrecc.jsonl', keep_in_memory=True)['train']

    template = """Please generate a question for the following answer: {0}, and the generated question should help search engine finding the following relevant passage: {1}.\
            \n\nGenerated question: """

    # prompt template (example)
    print(template.format("THIS IS AN EXAMPLE QUERY", "THIS IS AN EXAMPLE PASSGE"))

    # batch generate function
    def generate(generator, qids, texts, answers, rewrites, writer=None):
        results = generator.text_completion(
            texts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for i, (qid, result) in enumerate(zip(qids, results)):
            if writer is None:
                print(texts[i])
                print(">>")
                print(result['generation'])
                print("==============")
            else:
                writer.write(json.dumps({
                    'qid': qid, 
                    'Rewrite': rewrites[i],
                    'Answer': answers[i],
                    'Llama2_Rewrite': normalize(result['generation'])
                }, ensure_ascii=False)+'\n')

    # init generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    fout = open(filename, 'w') if filename is not None else None

    # iterate query-passage pair
    for batch in tqdm(
            batch_iterator(dataset, max_batch_size, return_index=False), 
            total=len(dataset)//max_batch_size+1
    ):
        ## input ( with prompt 
        batch_qids = batch['qid']
        batch_passages = batch['passage']
        batch_answers = batch['answer']
        batch_passages = [" ".join(p.split()[:384]) for p in batch_passages]
        batch_texts = [template.format(a, p) for \
                (a, p) in zip(batch_answers, batch_passages)]
        generate(generator, batch_qids, batch_texts, 
                batch_answers, batch['question'], writer=fout)

    fout.close()

if __name__ == "__main__":
    fire.Fire(main)
