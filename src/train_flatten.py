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
from data import DataCollatorForFunctionFlatten, get_qrecc_dataset
from models import FiDT5_flat
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
    model = FiDT5_flat.from_pretrained(model_args.model_name_or_path)

    # Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    ## Datacollator
    data_collator = DataCollatorForFunctionFlatten(
            tokenizer=tokenizer, 
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length,
            n_conversations=model_args.n_conversations,
            instruction_prefix=training_args.instruction_prefix,
            conversation_prefix=training_args.conversation_prefix,
            truncation=True,
            padding=True,
    )

    # Data
    ## Dataset
    dataset = get_qrecc_dataset(data_args.train_file)
    if training_args.do_eval:
        dataset['test'] = get_qrecc_dataset(
                data_args.eval_file
        )['train'].shuffle(seed=42).select(list(range(300)))
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
