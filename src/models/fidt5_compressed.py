import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import  Seq2SeqLMOutput, BaseModelOutput
from .fidt5_revised import FiDT5DecoderStack, FiDT5EncoderStackForCompressed

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FiDT5EncoderStackForCompressed(encoder_config, self.shared) # replace 

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FiDT5DecoderStack(decoder_config, self.shared)

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
        labels: Optional[torch.LongTensor] = None,
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
                    **kwargs
            )
        # expand attention mask # [NOTE] not sure if it works on eval.
        attention_mask = self._expand(attention_mask)

        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs, 
                labels=labels,
                **kwargs
        )

    def _expand(self, mask, n_conversations=None):
        n_conversations = n_conversations or self.encoder.n_conversations
        additional_mask = torch.ones(
                (mask.size(0), n_conversations), 
                device=mask.device
        )
        mask_expanded = torch.cat([mask, additional_mask], -1)
        return mask_expanded
