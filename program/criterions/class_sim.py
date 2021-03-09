import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('class_sim')
class class_sim(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('class_sim', criterion_config.hidden_dim, criterion_config.num_label)
    
    # def __init__(self, rep_dim, num_label):
    #     super().__init__('classification', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        source, target = extra_input['to_loss']
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
        return ret

    def corsim(self, rep, predict, src, targets):
        cen_rep = torch.cat([torch.mean(rep[torch.eq(predict, i)], dim = 0) for i in range(self.num_label)]).reshape([-1, self.num_label])
        cen_src = torch.cat([torch.mean(src[torch.eq(targets, i)], dim = 0) for i in range(self.num_label)]).reshape([-1, self.num_label])
        return self.mse(cen_rep, cen_src)
    
    def mse(self, vec1, vec2):
        v1 = vec1.unsqueeze(0).repeat(vec2.shape[0], 1, 1)
        v2 = vec2.unsqueeze(1).repeat(1, vec1.shape[0], 1)

        ret = (v2 - v1) ** 2
        ret = torch.mean(ret, dim = -1)

    @classmethod
    def setup_criterion(cls):
        return cls

    