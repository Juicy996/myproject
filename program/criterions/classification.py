import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('cem_spec')
class classification_enreopy_mean_spec(basic_criterion):    
    def __init__(self, rep_dim, num_label):
        super().__init__('classification_enreopy_mean_spec', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        rep = torch.mean(rep, dim = 1)
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

@register_criterion('cem_config')
class classification_enreopy_mean_config(basic_criterion):    
    def __init__(self, criterion_config):
        super().__init__('classification_enreopy_mean_config', criterion_config.hidden_dim, criterion_config.num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        rep = torch.mean(rep, dim = 1)
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

@register_criterion('ced_spec')
class classification_enreopy_direct_spec(basic_criterion):    
    def __init__(self, rep_dim, num_label):
        super().__init__('classification_enreopy_direct_spec', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        # logits = torch.clamp(logits, min = 1e-9)
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

@register_criterion('ced_config')
class classification_enreopy_direct_config(basic_criterion):    
    def __init__(self, criterion_config):
        super().__init__('classification_enreopy_direct_config', criterion_config.hidden_dim, criterion_config.num_label)

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
        return ret
    
    @classmethod
    def setup_criterion(cls):
        return cls

@register_criterion('cxd_spec')
class classification_expect_direct_spec(basic_criterion):    
    def __init__(self, rep_dim, num_label):
        super().__init__('classification_expect_direct_spec', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        logits = torch.softmax(logits, dim = -1)
        target = target.unsqueeze(1)
        target = torch.zeros(logits.shape, device = logits.device).scatter_(1, target, 1)
        loss = -torch.sum(target * torch.log(logits + 1e-10))

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

@register_criterion('cxd_config')
class classification_expect_direct_config(basic_criterion):    
    def __init__(self, criterion_config):
        super().__init__('classification_expect_direct_config', criterion_config.hidden_dim, criterion_config.num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        logits = torch.softmax(logits, dim = -1)
        loss = -torch.sum(target.unsqueeze(1) * torch.log(logits + 1e-10))

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

@register_criterion('cxm_spec')
class classification_expect_mean_spec(basic_criterion):    
    def __init__(self, rep_dim, num_label):
        super().__init__('classification_expect_mean_spec', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        rep = torch.mean(ret, dim = 1)
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        logits = torch.softmax(logits, dim = -1)
        loss = -torch.sum(target.unsqueeze(1) * torch.log(logits + 1e-10))

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

@register_criterion('cxm_config')
class classification_expect_mean_config(basic_criterion):    
    def __init__(self, criterion_config):
        super().__init__('classification_expect_mean_config', criterion_config.hidden_dim, criterion_config.num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        rep = torch.mean(rep, dim = 1)
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        logits = torch.softmax(logits, dim = -1)
        loss = -torch.sum(target.unsqueeze(1) * torch.log(logits + 1e-10))

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




















