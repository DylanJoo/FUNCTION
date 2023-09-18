import sys
import multiprocessing
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    Trainer
)
from datasets import load_dataset

# customized modules
from models import FiDT5_comp
from data import DataCollatorForFunctionCompressed, get_qrecc_dataset
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
    model = FiDT5_comp.from_pretrained(model_args.model_name_or_path)

    # Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    ## Datacollator
    data_collator = DataCollatorForFunctionCompressed(
            tokenizer=tokenizer, 
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length,
            max_src_conv_length=data_args.max_src_conv_length,
            n_conversations=model_args.n_conversations,
            instruction_prefix=training_args.instruction_prefix,
            conversation_prefix=training_args.conversation_prefix,
            truncation=True,
            padding=True,
    )

    # Data
    ## Dataset
    dataset = get_qrecc_dataset(data_args.train_file)
    n_examples = len(dataset['train'])
    if training_args.do_eval:
        if 'qrecc' in data_args.eval_file:
            # qrecc test
            dataset['test'] = get_qrecc_dataset(data_args.eval_file)['train']
            dataset['test'] = dataset['test'].filter(
                    lambda example: example['Conversation_source'] != 'trec'
            ).shuffle(seed=42).select(list(range(1000)))
        else:
            # ikat train
            from data import get_ikat_dataset
            dataset['test'] = get_ikat_dataset(data_args.eval_file)
    else:
        dataset['test'] = None

    # Trainer
    trainer = Trainer(
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
