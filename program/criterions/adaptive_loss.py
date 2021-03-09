import os, sys, time, torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import register_criterion, basic_criterion, ProjectedAdaptiveLogSoftmax
from collections import defaultdict

@register_criterion('adaptive_loss')
class adaptive_loss(nn.Module):
    def __init__(self, criterion_config):
        super().__init__()
        self.criterion_name = 'adaptive_loss'
        self.hidden_dim = criterion_config.hidden_dim
        self.num_label = criterion_config.num_label
        self.cutoffs = criterion_config.cutoffs
        print('Criterion [{}] has beed built.'.format(self.criterion_name))

        self.crit = ProjectedAdaptiveLogSoftmax(self.num_label, self.hidden_dim, 2 * self.hidden_dim, self.cutoffs, div_val = div_val)

    def forward(self, hidden, target):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''
        tgt_len = target.size(0)

        pred_hid = hidden[-tgt_len:]

        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

        ret = {}
        ret['loss'] = loss
        ret['correct'] = 0
        ret['total'] = 0
        ret['loss_detach'] = loss.detach()
        ret['task_name'] = 'netgram'
        return ret

    @classmethod
    def setup_criterion(cls):
        return cls

    