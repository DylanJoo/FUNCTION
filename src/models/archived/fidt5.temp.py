import copy
import torch
from transformers import T5ForConditionalGeneration, T5Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import  Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Stack

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FiDT5Stack(encoder_config, self.shared) # replace 

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # add an adapter layer for adopting GTR embeddings
        ## [NOTE] the weights are random initialized
        self.embed_size_per_head = config.d_model // config.num_heads
        # Linear [H n_layer*H]
        self.proj_star = nn.Linear(
                config.d_model,
                config.num_decoder_layers * config.d_model,
                bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_crossattention_scores(self, context_mask):
        raise NotImplementedError('Please implement this function.')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_conv: Optional[torch.LongTensor] = None,
        attention_mask_conv: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        :param input_ids: the input for focal request.
        :param attention_mask: the mask for focal request.
        :param input_ids_conv: tokenized input_ids for (multi-turn) conversations.
        :param attention_mask_conv: the mask for (multi-turn) conversations.
        """

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    pass
            )

        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs, 
                **kwargs
        )

class FiDT5Stack(T5Stack):

    def forward(self, 
                input_ids, attention_mask, 
                input_ids_conv=None, attention_mask_conv=None,
                **kwargs):
        """ 
        Wrap/unwrap input/ouput with this class (replace t5-encoder) 

        :param input_ids: the tokenized input ids with shape (BN, L)
        :param attention_mask: the attention mask with shape (B, NL)
        :param input_ids_instruction: the input for focal request. (B, L)
        :param attention_mask_instruction: the mask for focal request. (B L)

        :return encoder_outputs: the huggingface model output class.
        """
        # FUNCTION: FUsion-iN-ConversaTION
        ## [REQUEST] tokenized input ids
        if input_ids.dim() == 3: # normal usage of FiD
            B, N, L = input_ids.size()
        else:
            B, L = input_ids.size()
            N = 1
        encoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                **kwargs
        )

        ## [CONV] tokenized input ids for multi-turn conversation
        if input_ids_conv.dim() == 3: # normal usage of FiD
            B, N, L = input_ids_conv.size()
        else:
            B, L = input_ids_conv.size()
            N = 1
        input_ids_conv = input_ids_conv.view(B*N, -1)
        attention_mask_conv = attention_mask_conv.view(B*N, -1)
        encoder_outputs_conv = super().forward(
                input_ids=input_ids_conv,
                attention_mask=attention_mask_conv, 
                **kwargs
        )

        ## [MERGE] combine the token-level 
        encoder_outputs['attention_mask']
        encoder_outputs_conv['last_hidden_state'] = \
                encoder_outputs_conv['last_hidden_state'].view(B, N*L, -1)

        return encoder_outputs

