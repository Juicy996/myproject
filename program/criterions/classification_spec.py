import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('classification_spec')
class classification_spec(basic_criterion):    
    def __init__(self, rep_dim, num_label):
        super().__init__('classification_spec', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        # loss = torch.nn.functional.nll_loss(logits, target, size_average = False, ignore_index = self.padding_idx, reduce = reduce)
        loss = torch.nn.functional.cross_entropy(logits, target)#, reduction = 'sum', ignore_index = self.padding_idx

        ret = {}
        ret['loss'] = loss.unsqueeze(0)
        ret['correct'] = corrects.detach().unsqueeze(0)
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype).detach().unsqueeze(0)
        ret['loss_detach'] = loss.detach().unsqueeze(0)
        ret['rep'] = logits.detach()
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

    