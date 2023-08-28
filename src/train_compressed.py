import sys
import multiprocessing
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset

# customized modules
from data import DataCollatorForStarter
from trainers import TrainerForStarter
from models import FiDT5
from arguments import ModelArgs, DataArgs, TrainArgs

import os

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = \
                parser.parse_args_into_dataclasses()

    # Preparation 
    # (tokenizer, prompt indices)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Model
    model = FiDT5.from_pretrained(model_args.model_name_or_path)

    # Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    ## Datacollator
    data_collator = DataCollatorForStarter(
            retrieval_enhanced=model_args.retrieval_enhanced,
            tokenizer=tokenizer, 
            max_src_length=data_args.max_p_length,
            max_tgt_length=data_args.max_q_length,
            truncation=True,
            padding=True,
            sep_token='</s>',
            star_encoder=star_encoder
    )

    # Data
    ## Dataset
    dataset = load_dataset('json', data_files=data_args.train_file)
    dataset = dataset.map(lambda ex: {
        'statement_aware_embeds': torch.tensor(ex['statement_aware_embeds'])
    })
    n_examples = len(dataset['train'])
    if training_args.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Trainer
    trainer = TrainerForStarter(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
