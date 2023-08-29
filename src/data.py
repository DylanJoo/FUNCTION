from copy import copy
import random
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
    PaddingStrategy, 
    PreTrainedTokenizerBase
)
from datasets import load_dataset, Dataset

def get_qrecc_dataset(path):
    def split_context(context):
        context_list = []
        while len(context) >= 2:
            context_list.append((context.pop(0), context.pop(0)))
        return context_list

    dataset = load_dataset('json', data_files=path)
    dataset = dataset.map(
            lambda x: {'Conversation': split_context(x['Context'])}
    )
    dataset = dataset.map(lambda ex: {"id": f"{ex['Conversation_source']}_{ex['Conversation_no']}_{ex['Turn_no']}"})

    return dataset

def get_ikat_dataset(path):
    dataset = load_dataset('json', data_files=path)['train']

    # flatten the turns
    data_list = []
    for topic in dataset:
        topic_id = topic['number']
        try:
            # [NOTE] `Dataset` would make the dict length consistent, 
            # so it'll add None
            ptkbs = {k: v for k, v in topic['ptkb'].items() if v is not None}
        except:
            continue

        history = []
        for turn in topic['turns']:
            data = {}

            # turn
            turn_id = turn['turn_id']
            utterance = turn['utterance']
            response = turn['response']

            # collect data
            data['id'] = f"{topic_id}_{turn_id}" 

            ## qrecc: question / conversations/ rewrite
            data['Question'] = utterance
            data['Conversation'] = copy(history)
            data['Rewrite'] = turn['resolved_utterance']
            data['selected_ptkbs'] = [\
                    ptkbs[str(i)] for i in turn['ptkb_provenance']\
            ]

            ## use all ptkbs
            data['all_ptkbs'] = random.sample(
                    list(ptkbs.values()), k=len(list(ptkbs.values()))
            )
            data_list.append(data)

            ## historical utterances
            history.append([utterance, response])

    return Dataset.from_list(data_list)

@dataclass
class DataCollatorForFunctionFlatten:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_src_length: Optional[int] = 288
    max_tgt_length: Optional[int] = 32
    n_conversations: Optional[int] = 1
    instruction_prefix: Optional[str] = 'Rewrite the request according to the user-system conversation. Request: {} Conversation: '
    conversation_prefix: Optional[str] = 'user: {0} sytem: {1}'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ The expected input would be:

        Source input:
        Rewrite the request according to the user-system conversation. \
                request: {request} \
                conversation: user: {utterance} system: {response}

        Target output:
        {rewritten query}
        """

        # preparing source/target
        sources, targets = [], []
        for batch in features:
            ## request (focal question/utterance)
            utterance = batch['Question']
            ## user-system conversation # reverse it (the last n turns)
            avail_conversation = batch['Conversation'][-self.n_conversations:]
            avail_conversation += [["<pad>", "<pad>"]] * self.n_conversations

            ## Conversation loop
            for i, conversation in enumerate(avail_conversation[:self.n_conversations]):
                if i == 0:
                    sources.append(
                            self.instruction_prefix.format(utterance)
                    )
                sources.append(
                        self.conversation_prefix.format(*conversation)
                )
            ## rewritten questions
            targets.append(batch['Rewrite'])

        # tokenizing src/tgt
        inputs = self.tokenizer(
                sources,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )
        inputs['labels'] = self.tokenizer(
                targets,
                max_length=self.max_tgt_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        ).input_ids

        # postprocess
        ## - input_ids: (BN, L)
        ## - attention_mask: (B, NL)
        ## - labels: (B, L_tgt)
        inputs['input_ids'] = inputs['input_ids'].view(
                -1, (1+self.n_conversations), inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
                -1, (1+self.n_conversations)*inputs['attention_mask'].size(-1)
        )
        return inputs

@dataclass
class DataCollatorForFunctionCompressed:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_src_length: Optional[int] = 64
    max_tgt_length: Optional[int] = 32
    max_src_conv_length: Optional[int] = 128
    n_conversations: Optional[int] = 1
    instruction_prefix: Optional[str] = 'Rewrite the request according to the user-system conversation. Request: {} Conversation: '
    conversation_prefix: Optional[str] = 'user: {0} sytem: {1}'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ The expected input would be:

        Source input:
        Rewrite the request according to the user-system conversation. \
                request: {request} \
                conversation: user: {utterance} system: {response}

        Target output: {rewritten query}
        """

        # preparing source/target
        sources, sources_conv, targets = [], [], []
        for batch in features:
            ## Utterance
            sources.append(self.instruction_prefix.format(batch['Question']))

            ## Historical conversations
            avail_conversation = batch['Conversation'][-self.n_conversations:]
            avail_conversation += [["<pad>", "<pad>"]] * self.n_conversations
            for i, conversation in enumerate(avail_conversation[:self.n_conversations]):
                sources_conv.append(
                        self.conversation_prefix.format(*conversation)
                )

            ## Rewritten
            targets.append(batch['Rewrite'])

        # tokenizing src1/src2/tgt
        ## utterances/conversations/rewritten
        inputs = self.tokenizer(
                sources,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )

        inputs_conv = self.tokenizer(
                sources_conv,
                max_length=self.max_src_conv_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )
        inputs['input_ids_conv'] = inputs_conv['input_ids']
        inputs['attention_mask_conv'] = inputs_conv['attention_mask']

        inputs['labels'] = self.tokenizer(
                targets,
                max_length=self.max_tgt_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        ).input_ids

        return inputs

@dataclass
class DataCollatorForNTR:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_src_length: Optional[int] = 512
    max_tgt_length: Optional[int] = 32
    n_conversations: Optional[int] = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Usage:
        Source input: {utterance} ||| {response} ||| {request}
        Target output: {rewritten query}
        """
        # preparing source/target
        sources, targets = [], []
        for batch in features:
            history = []
            ## Focal question, utterance
            utterance = batch['Question']
            ## user-system conversation 
            avail_conversation = batch['Conversation']

            i = 0
            while i < min(len(avail_conversation), self.n_conversations):
                conversation = avail_conversation.pop(0)

                if len(conversation) == 1: 
                    # ptkbs
                    history.append(conversation[0]) 
                else:
                    # user system conversation
                    history.append(conversation[0])
                    history.append(conversation[1])
                    i += 1

            sources.append(" ||| ".join( history + [utterance] ))
            targets.append(batch['Rewrite'])

        # tokenizing src/tgt
        inputs = self.tokenizer(
                sources,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )
        inputs['labels'] = self.tokenizer(
                targets,
                max_length=self.max_tgt_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        ).input_ids

        return inputs
