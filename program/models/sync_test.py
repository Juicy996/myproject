import torch
from . import basic_model, register_model
from .. import modules
import torch.nn as nn
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

@register_model('sync_test')
class sync_test(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'sync_test', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.linear1 = torch.nn.Linear(300, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 300)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = source

        vec = self.linear1(vec)
        vec = torch.relu(vec)
        vec = self.linear2(vec)
        vec = torch.relu(vec)
        vec = self.linear3(vec)
        vec = torch.relu(vec)
        vec = self.linear4(vec)
        vec = torch.relu(vec)

        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls