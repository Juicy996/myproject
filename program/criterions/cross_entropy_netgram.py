import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('netgram')
class cross_entropy_netgram(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('netgram', criterion_config.hidden_dim, criterion_config.num_label)

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
        loss = torch.nn.functional.cross_entropy(logits, target).unsqueeze(0)


        # nll = -torch.nn.functional.log_softmax(logits, dim = -1)
        # loss = torch.gather(nll.reshape([-1, self.num_label]), dim = 1, index = target.long().reshape([-1, 1]))


        # loss = loss / len(target)
        ret = {}
        ret['loss'] = loss.unsqueeze(0)
        ret['correct'] = corrects.detach().unsqueeze(0)
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype).unsqueeze(0)
        ret['loss_detach'] = loss.detach().unsqueeze(0)
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

    