import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
    PaddingStrategy, 
    PreTrainedTokenizerBase
)
from datasets import load_dataset

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
    return dataset

@dataclass
class DataCollatorForFunctionFlatten:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_src_length: Optional[int] = 288
    max_tgt_length: Optional[int] = 32
    n_conversations: Optional[int] = 1
    instruction_prefix: Optional[str] = 'Rewrite the request according to the user-system conversation. request: {} conversation: '
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
            avail_conversation = batch['Conversation'][::-1]
            avail_conversation += [["<pad>", "<pad>"]] * self.n_conversations
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
    instruction_prefix: Optional[str] = 'Rewrite the request according to the user-system conversation. request: {} conversation: '
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
        sources, sources_conv, targets = [], [], []
        for batch in features:
            ## Utterance
            sources.append(self.instruction_prefix.format(batch['Question']))

            ## Historical conversations
            avail_conversation = batch['Conversation'][::-1]
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
        # inputs['input_ids_conv'] = inputs_conv['input_ids'].view(
        #         -1, self.n_conversations, inputs['input_ids'].size(-1)
        # )