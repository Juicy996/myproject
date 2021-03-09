import os, sys, torch
from . import register_optimizer
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

@register_optimizer('adabelief')
class adabelief(AdaBelief):
    def __init__(self, params, optimizer_config):
        super().__init__(params, 
        lr = optimizer_config.lr, 
        weight_decay = optimizer_config.weight_decay,
        eps=1e-16, betas=(0.9,0.999), 
        weight_decouple = True,
        rectify = True,)
        print('Optimizer [{}] has beem built.'.format('Adam'))

    @classmethod
    def setup_optimizer(cls):
        return cls

@register_optimizer('rangeradabelief')
class RangerAdaBelief(RangerAdaBelief):
    def __init__(self, params, optimizer_config):
        super().__init__(params, 
        lr = optimizer_config.lr, 
        weight_decay = optimizer_config.weight_decay,
        eps=1e-12, 
        betas=(0.9,0.999),)
        print('Optimizer [{}] has beem built.'.format('Adam'))

    @classmethod
    def setup_optimizer(cls):
        return cls

















