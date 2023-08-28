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

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

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
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_ids_conv=input_ids_conv, 
                    attention_mask_conv=attention_mask_conv,
            )

        # expand attention mask
        attenion_mask = self._expand(attention_mask, input_ids_conv.shape[0])

        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs, 
                **kwargs
        )

    def _expand(self, mask, n_conversations=1):
        additional_mask = torch.ones(
                (mask.size(0), n_conversations), 
                device=mask.device
        )
        mask = torch.cat([additional_mask, mask], -1)
        return mask

class FiDT5Stack(T5Stack):

    def forward(self, 
                input_ids, attention_mask, 
                input_ids_conv, attention_mask_conv,
                **kwargs):
        """ 
        FUNCTION: FUsion-iN-ConversaTION

        :param input_ids: the input for focal request. (B, L)
        :param attention_mask: the mask for focal request. (B L)
        :param input_ids_conv: the tokenized input ids of conversations.
        :param attention_mask_conv: the attention mask of conversations.
        """
        ## Sizes
        B = input_ids.size(0)
        N = input_ids_conv.size(0) // B
        L = input_ids_conv.size(1)

        ## Utterances 
        encoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                **kwargs
        )
        print(encoder_outputs['last_hidden_state'].shape)

        ## Conversations
        encoder_outputs_conv = super().forward(
                input_ids=input_ids_conv,
                attention_mask=attention_mask_conv, 
                **kwargs
        )
        ### Convert conversational token embeddings
        ### into conversational sentence embeddins
        ### B N L H  --> B N H (mean embeddings)
        conversation_embeds = \
                encoder_outputs_conv['last_hidden_state'].view(B, N, L, -1)
        conversation_attn_mask = attention_mask_conv.view(B, N, L)
        compressed_embeds = self.mean_pooling(
                conversation_embeds, conversation_attn_mask, 2
        ) 
        print(conversation_embeds.shape)

        ## [MERGE] combine the token-level 
        encoder_outputs_conv['last_hidden_state'] = torch.cat([
            encoder_outputs['last_hidden_state'], 
            compressed_embeds
        ], dim=1)
        print(encoder_outputs_conv['last_hidden_state'].shape)

        return encoder_outputs

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask, dim=1):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=dim) / torch.clamp(input_mask_expanded.sum(dim), min=1e-9)
