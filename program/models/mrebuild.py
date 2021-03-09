import torch
from . import basic_model, register_model
from .. import modules
import torch.nn as nn
if True:
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    except:
        from torch.nn import LayerNorm
else:
    from torch.nn import LayerNorm

@register_model('mrebuild')
class mrebuild(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'mrebuild', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.task_type = 'rebuild'

        self.net = nn.ModuleList()
        for i in range(model_config.nlayer): 
            self.net.append(layer(self.hidden_dim))
        
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
        self.dropi = nn.Dropout(model_config.dropp)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        if isinstance(embedding, nn.ModuleDict):
            vec = embedding['emb_src'](source.long())
            wiz = embedding['emb_tgt'](wizard.long())
            vec = torch.cat([wiz, vec], dim = 1)
        else:
            vec = embedding(source.long())  

        for l in self.net:
            vec = l(vec)

        vec = torch.cat([ch, cc, dh[:, None, :]], dim = 1)
        
        vec = self.ff(vec)

        vec = self.pool(vec)
        vec = self.fnn(vec).reshape([-1, 300])
        vec = self.dropi(vec)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls





























