import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='prediction.jsonl')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            print(data['qid'], data['question'])
            print(data['generated_question'])
            print(data['rewritten_question'])

