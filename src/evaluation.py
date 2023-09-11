import json
import string
from tqdm import tqdm
import evaluate
import argparse
from datasets import load_dataset

def get_score(evaluate, predictions, references, metric_key=None, **kwargs):
    results = evaluate.compute(
            predictions=predictions, 
            references=references,
            **kwargs
    )

    if metric_key is None: # return the first key
        return results
    else:
        return {metric_key: results[metric_key]}

def extract_diff_tokens(x, y):
    x = x.translate(
            str.maketrans('', '', string.punctuation)
    ).split()
    y = y.translate(
            str.maketrans('', '', string.punctuation)
    ).lower()
    return " ".join([tok for tok in x if tok.lower() not in y])

def load_file(input_file, baseline=False, diff_as_reference=False):
    dataset = load_dataset(
            'json', data_files=input_file, keep_in_memory=True
    )['train']

    if baseline:
        predictions = dataset['question']
    else:
        predictions = dataset['generated_question']

    if diff_as_reference:
        references = dataset.map(lambda x: 
                {'rewritten_tokens': extract_diff_tokens(
                    x['rewritten_question'], x['question']
                )})['rewritten_tokens']
    else:
        references = dataset['rewritten_question']

    return predictions, references

    # predictions, references = [], []
    # with open(input_file, 'r') as f:
    #     for line in tqdm(f):
    #         data = json.loads(line.strip())
    #         # use the baseline as prediction
    #         if baseline:
    #             predictions.append(data['question'])
    #         else:
    #             predictions.append(data['generated_question'])
    #
    #         # use new tokens as reference
    #         if diff_as_reference:
    #             baseline_toks = data['question'].translate(
    #                     str.maketrans('', '', string.punctuation)
    #             ).lower().split()
    #             reference_toks = data['rewritten_question'].translate(
    #                     str.maketrans('', '', string.punctuation)
    #             ).split()
    #             reference_toks_diff = \
    #                     [tok for tok in reference_toks \
    #                     if tok.lower() not in baseline_toks]
    #             references.append(" ".join(reference_toks_diff))
    #         else:
    #             references.append(data['rewritten_question'])
    #
    # return predictions, references


def main(input_file, metrics, baseline=False, diff_as_reference=False):
    # load
    predictions, references = \
            load_file(input_file, baseline, diff_as_reference)

    # compute
    results = {}
    print('# used metrics:', metrics)

    if 'bleu' in " ".join(metrics):
        scores = get_score(
                evaluate.load('bleu'), 
                predictions,
                references
        )
        results.update(scores)
    if 'rouge' in " ".join(metrics):
        scores = get_score(
                evaluate.load('rouge'), 
                predictions,
                references
        )
        results.update(scores)
    if 'bert' in " ".join(metrics):
        scores = get_score(
                evaluate.load('bertscore'),
                predictions, 
                references,
                lang='en',
                device='cuda'
        )
        for k in ['precision', 'recall', 'f1']: 
            results.update({f"bertscore-{k}": sum(scores[k])/len(scores[k])})

    for metric in metrics:
        if ('rouge' not in metric) and ('bleu' not in metric) and ('bert' not in metric):
            scores = get_score(
                    evaluate.load(metric),
                    predictions, 
                    references,
            )
            results.update(scores)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", action='append', default=[])
    parser.add_argument("--jsonl", type=str, default='prediction.jsonl')
    parser.add_argument("--baseline", action='store_true', default=False)
    parser.add_argument("--diff_as_reference", action='store_true', default=False)
    args = parser.parse_args()

    results = main(
            input_file=args.jsonl,
            metrics=args.metric,
            baseline=args.baseline,
            diff_as_reference=args.diff_as_reference
    )

    # print
    for k, v in results.items():
        try:
            print(k, round(v, 4))
        except:
            print(k, v)
