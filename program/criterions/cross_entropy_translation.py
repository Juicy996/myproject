import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('translation')
class cross_entropy_translation(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('translation', criterion_config.hidden_dim, criterion_config.num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        # input(logits)
        logits = logits.reshape([-1, self.num_label])
        
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        # input(predicts)
        
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        # loss = torch.nn.functional.nll_loss(logits, target, size_average = False, ignore_index = self.padding_idx, reduce = reduce)
        loss = torch.nn.functional.cross_entropy(logits, target)#, reduction = 'sum', ignore_index = self.padding_idx

        ret = {}
        ret['loss'] = loss
        ret['correct'] = corrects.detach()
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype)
        ret['loss_detach'] = torch.tensor(loss.detach())
        ret['task_name'] = torch.tensor(2, device = loss.device, dtype = self.fnn[0].weight.dtype)
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

    