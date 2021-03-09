import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('trainable_mse')
class trainable_mse(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('trainable_mse', criterion_config.hidden_dim, criterion_config.num_label)
        classes = torch.FloatTensor(self.num_label, self.hidden_dim)
        torch.nn.init.xavier_uniform_(classes)
        self.classes = torch.nn.Parameter(classes)

        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(in_features  = self.hidden_dim, out_features  = 2 * self.hidden_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features  = 2 * self.hidden_dim, out_features  = self.hidden_dim)
        ) 

    def forward(self, rep, target, reduce = True, extra_input = None):
        batch_size = rep.shape[0]
        logits = self.fnn(rep)

        logits = logits.unsqueeze(1).repeat(1, self.num_label, 1)
        wizlab = self.classes.unsqueeze(0).repeat(batch_size, 1, 1)

        mses = torch.mean((wizlab - logits) ** 2, dim = -1)
        predicts = torch.min(mses.detach(), dim = 1)[1]
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        gloss = torch.gather(mses, 1, target.reshape([-1, 1]).long())
        loss = torch.sum(gloss)

        ret = {}
        ret['loss'] = loss.unsqueeze(0)
        ret['correct'] = corrects.detach().unsqueeze(0)
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype).detach().unsqueeze(0)
        ret['loss_detach'] = loss.detach().unsqueeze(0)
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

@register_criterion('trainable_cos')
class trainable_cos(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('classification', criterion_config.hidden_dim, criterion_config.num_label)
        classes = torch.FloatTensor(self.num_label, self.hidden_dim)
        torch.nn.init.xavier_uniform_(classes)
        self.classes = torch.nn.Parameter(classes)

        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(in_features  = self.hidden_dim, out_features  = 2 * self.hidden_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features  = 2 * self.hidden_dim, out_features  = self.hidden_dim)
        ) 

    def forward(self, rep, target, reduce = True):
        batch_size = rep.shape[0]
        logits = self.fnn(rep)

        logits = logits.unsqueeze(1).repeat(1, self.num_label, 1)
        wizlab = self.classes.unsqueeze(0).repeat(batch_size, 1, 1)

        x2 = torch.sum(logits ** 2, dim = -1)
        y2 = torch.sum(wizlab ** 2, dim = -1)
        cos = -(torch.sum(logits * wizlab, dim = -1) / (x2 ** 0.5 * y2 ** 0.5))

        predicts = torch.min(cos.detach(), dim = 1)[1]
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        gloss = torch.gather(cos, 1, target.reshape([-1, 1]))
        loss = torch.mean(gloss)

        ret = {}
        ret['loss'] = loss
        ret['correct'] = corrects.detach()
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype).detach()
        ret['loss_detach'] = loss.detach()
        ret['task_name'] = torch.tensor(1, device = loss.device, dtype = self.fnn[0].weight.dtype).detach()
        # input(ret)
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

@register_criterion('trainable_l1')
class trainable_l1(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('classification', criterion_config.hidden_dim, criterion_config.num_label)
        classes = torch.FloatTensor(self.num_label, self.hidden_dim)
        torch.nn.init.xavier_uniform_(classes)
        self.classes = torch.nn.Parameter(classes)

        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(in_features  = self.hidden_dim, out_features  = 2 * self.hidden_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features  = 2 * self.hidden_dim, out_features  = self.hidden_dim)
        ) 

    def forward(self, rep, target, reduce = True):
        batch_size = rep.shape[0]
        logits = self.fnn(rep)

        logits = logits.unsqueeze(1).repeat(1, self.num_label, 1)
        wizlab = self.classes.unsqueeze(0).repeat(batch_size, 1, 1)

        t = torch.abs(wizlab - logits)
        t = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        l1 = torch.mean(t, dim = -1)

        predicts = torch.min(l1.detach(), dim = 1)[1]
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        gloss = torch.gather(l1, 1, target.reshape([-1, 1]))
        loss = torch.sum(gloss)

        ret = {}
        ret['loss'] = loss
        ret['correct'] = corrects.detach()
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype).detach()
        ret['loss_detach'] = loss.detach()
        ret['task_name'] = torch.tensor(1, device = loss.device, dtype = self.fnn[0].weight.dtype).detach()
        # input(ret)
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

























    