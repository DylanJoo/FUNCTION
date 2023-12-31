import torch
import torch.nn as nn
import torch.nn.functional as F

def InBatchKLLoss(logits_pred, logits_truth, temperature=0.25):
    KLLoss = nn.KLDivLoss(reduction='batchmean')
    y_pred = F.log_softmax(logits_pred, dim=1) # usually log the logits
    y_true = F.softmax(logits_truth/temperature, dim=1) # soft label (not log-styled)
    return KLLoss(y_pred, y_true)

def InBatchNegativeCELoss(logits):
    CELoss = nn.CrossEntropyLoss()
    labels = torch.arange(0, logits.size(0), device=logits.device)
    return CELoss(logits, labels)

def PairwiseCELoss(scores, pos_idx=1):
    CELoss = nn.CrossEntropyLoss()
    logits = scores.view(2, -1).permute(1, 0) # (B*2 1) -> (B 2)
    labels = torch.tensor(
            [pos_idx]*logits.shape[0], dtype=torch.long, device=logits.device
    )
    return CELoss(logits, labels)

def LMCELoss(logits, labels, vocab_size):
    CELoss = nn.CrossEntropyLoss()
    logits = logits.view(-1, vocab_size)
    labels = labels.to(logits.device).view(-1)
    return CELoss(logits, labels)

