import os, sys, torch
from . import register_optimizer


@register_optimizer('adam')
class adam(torch.optim.Adam):
    def __init__(self, params, optimizer_config):
        super().__init__(params, 
        lr = optimizer_config.lr, 
        weight_decay = 0)
        # weight_decay = optimizer_config.weight_decay)
        print('Optimizer [{}] has beem built.'.format('Adam'))

    @classmethod
    def setup_optimizer(cls):
        return cls























