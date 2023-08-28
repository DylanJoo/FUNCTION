import argparse
import time
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tools import (
    load_ikat_topics,
    load_topics,
)

def rewrite(texts, max_length, **gen_config):

    input_ids = tokenizer(texts, 
                          return_tensors="pt", 
                          padding=True,
                          truncation=True).to(model.device)

    outputs = model.generate(
            **input_ids, 
            max_length=args.max_length, 
            **gen_config
    )
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True,)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="castorini/t5-base-canard", type=str)
    parser.add_argument("--query", default="sample.jsonl", type=str)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--concat_ptkb", default=False, action='store_true')
    parser.add_argument("--output", default="sample.tsv", type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    args = parser.parse_args()

    # load hf 
    model_name = args.model_name_or_path
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(args.device)
    tokenizer = T5Tokenizer.from_pretrained(model_name, truncation_side="left")

    # load topics
    if "ikat" in args.query.lower():
        query = load_ikat_topics(
                args.query, resolved=resolved, concat_ptkb=concat_ptkb
        )
    elif 'qrecc' in args.query_rewritten.lower():
        query = load_qrecc_topics(args.query)
    else:
        query = load_topics(args.query)

    ikat_topic = load_ikat_topics(args.input, args.concat_ptkb)

    with open(args.output, 'r') as fin, open(args.output, 'w') as fout:

        for line in fin:
            data_dict = json.loads(line.strip())

            # extract data dict information
            topic_turn_id = data_dict['topic_turn_id']
            utterance = data_dict['utterance']
            history_utterances = data_dict['history_utterances']
            history_responses = data_dict['history_responses']

            # prepare context
            n_history = len(history_utterances)
            n_selected = min(3, n_history)

            context_ur = [f"{u} ||| {r}" for u, r in \
                    zip(history_utterances[-n_selected:], history_responses[-n_selected:])]
            if n_history <= n_selected:
                ## case1: history depth less than k (3)
                context_u = []
            else:
                ## case2: history depth greater than k (3)
                context_u = [f"{u}" for u in history_utterances[:-n_selected]]

            context = " ||| ".join(context_u + context_ur + [utterance])
            query_rewritten = rewrite(context)

            fout.write(f"{topic_turn_id}\t{query_rewritten}\n")
