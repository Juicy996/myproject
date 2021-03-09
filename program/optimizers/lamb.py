import os, sys, torch
from pytorch_lamb import Lamb
from . import register_optimizer


@register_optimizer('lamb')
class adam(Lamb):
    def __init__(self, params, optimizer_config):
        super().__init__(
            params = params, 
            lr = optimizer_config.lr, 
            weight_decay = 0,
            # weight_decay = optimizer_config.weight_decay,
            min_trust = optimizer_config.min_trust,
        )
        print('Optimizer [{}] has beem built.'.format('lamb'))

    @classmethod
    def setup_optimizer(cls):
        return cls























