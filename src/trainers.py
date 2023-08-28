import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
from transformers import (
    Trainer, 
    Seq2SeqTrainer,
    PreTrainedModel
)
from loss import InBatchNegativeCELoss as info_nce
from loss import PairwiseCELoss as pair_ce
from loss import LMCELoss as gen_ce

class TrainerForStarter(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model.forward(**inputs)

        # Calculate losses
        ## generation NLL/CE loss
        loss = outputs.get('loss', 0)
        # loss = gen_ce(
        #         outputs['logits'], inputs['labels'], 
        #         model.config.vocab_size
        # )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
