import json
from tqdm import tqdm
import evaluate
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", action='append', default=[])
    parser.add_argument("--jsonl", type=str, default='prediction.jsonl')
    parser.add_argument("--baseline", action='store_true', default=False)
    args = parser.parse_args()

    predictions = []
    references = []
    with open(args.jsonl, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            if args.baseline:
                predictions.append(data['question'])
            else:
                predictions.append(data['generated_question'])
            references.append(data['rewritten_question'])

        results = {}
        print('# used metrics:', args.metric)
        if 'bleu' in " ".join(args.metric):
            scores = get_score(
                    evaluate.load('bleu'), 
                    predictions,
                    references
            )
            results.update(scores)

        if 'rouge' in " ".join(args.metric):
            scores = get_score(
                    evaluate.load('rouge'), 
                    predictions,
                    references
            )
            results.update(scores)
        if 'bert' in " ".join(args.metric):
            scores = get_score(
                    evaluate.load('bertscore'),
                    predictions, 
                    references,
                    lang='en',
                    device='cuda'
            )
            for k in ['precision', 'recall', 'f1']: 
                results.update({f"bertscore-{k}": sum(scores[k])/len(scores[k])})

        for metric in args.metric:
            if ('rouge' not in metric) and ('bleu' not in metric) and ('bert' not in metric):
                scores = get_score(
                        evaluate.load(metric),
                        predictions, 
                        references,
                )
                results.update(scores)

        # output
        for k, v in results.items():
            try:
                print(k, round(v, 4))
            except:
                print(k, v)
