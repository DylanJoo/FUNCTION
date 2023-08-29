import json
import copy
import torch
import argparse
import collections
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    T5ForConditionalGeneration
)
from models import FiDT5_flat, FiDT5_comp
from data import (
    DataCollatorForFunctionFlatten, 
    DataCollatorForFunctionCompressed,
    DataCollatorForNTR,
    get_qrecc_dataset
)

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
    parser.add_argument("--input_file")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--instruction_prefix", default=None, type=str)
    parser.add_argument("--conversation_prefix", default=None, type=str)
    parser.add_argument("--n_conversations", default=1, type=int)
    parser.add_argument("--output_history", default=False, action='store_true')

    # generation config
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_src_length", default=128, type=int)
    parser.add_argument("--max_tgt_length", default=32, type=int)
    parser.add_argument("--max_src_conv_length", default=128, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    # Model
    ## config and tokenizers
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ## checkpoints and }atacollator
    if 'flatten' in args.model_path.lower():
        model = FiDT5_flat.from_pretrained(args.model_path)
        data_collator = DataCollatorForFunctionFlatten(
                tokenizer=tokenizer, 
                max_src_length=args.max_src_length,
                max_tgt_length=args.max_tgt_length,
                n_conversations=args.n_conversations,
                instruction_prefix=args.instruction_prefix,
                conversation_prefix=args.conversation_prefix
        )
    if 'compressed' in args.model_path.lower():
        model = FiDT5_comp.from_pretrained(args.model_path)
        data_collator = DataCollatorForFunctionCompressed(
                tokenizer=tokenizer, 
                max_src_length=args.max_src_length,
                max_tgt_length=args.max_tgt_length,
                max_src_conv_length=args.max_src_conv_length,
                n_conversations=args.n_conversations,
                instruction_prefix=args.instruction_prefix,
                conversation_prefix=args.conversation_prefix
        )
    if 'castorini' in args.model_path.lower():
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        data_collator = DataCollatorForNTR(
                tokenizer=tokenizer,
                max_src_length=args.max_src_length,
                max_tgt_length=args.max_tgt_length,
                n_conversations=args.n_conversations
        )
    model.to(args.device)
    model.eval()


    # Data
    dataset = get_qrecc_dataset(args.input_file)['train'].to_list()
    dataset_iter = batch_iterator(
            dataset, size=args.batch_size, return_index=False
    )

    # Generation
    with torch.no_grad(), open(args.output_jsonl, 'w') as fout:
        for batch_features in tqdm(dataset_iter):
            batch_inputs = data_collator(batch_features)
            batch_inputs = batch_inputs.to(args.device)

            # no labels
            batch_inputs.pop('labels')
            output_ids = model.generate(
                    **batch_inputs,
                    num_beams=args.num_beams,
                    max_length=args.max_tgt_length,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
            )

            output_texts = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
            )

            for i, feature in enumerate(batch_features):
                if args.output_history:
                    fout.write(json.dumps({
                        "qid": feature['id'],
                        "question": feature['Question'],
                        "generated_question": output_texts[i],
                        "rewritten_question": feature['Rewrite'],
                        "history": feature['Conversation']
                    }) + '\n')
                else:
                    fout.write(json.dumps({
                        "qid": feature['id'],
                        "question": feature['Question'],
                        "generated_question": output_texts[i],
                        "rewritten_question": feature['Rewrite'],
                    }) + '\n')

    print('done')


