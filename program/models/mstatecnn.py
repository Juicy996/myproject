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

@register_model('mstatecnn')
class mstatecnn(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'mstatecnn', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        percom = self.hidden_dim
        nlayer = model_config.nlayer
        self.net = nn.ModuleList()
        for i in range(nlayer): 
            self.net.append(layer(self.hidden_dim, percom))
        
        self.ff = nn.Sequential(
            nn.Linear(percom * nlayer * 28, 2 * self.hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
        self.dropi = nn.Dropout(model_config.dropp)
        
        self.fpool = None
        # self.fpool = modules.FlexiblePatt(28, [1, 2, 3, 4, 5, 6, 7], percom * nlayer)
        self.fpool = modules.FlexibleLKmul(28, percom * nlayer)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        if embedding:
            if isinstance(embedding, nn.ModuleDict):
                vec = embedding['emb_src'](source.long())
                wiz = embedding['emb_tgt'](wizard.long())
                vec = torch.cat([wiz, vec], dim = 1)
            else:
                vec = embedding(source.long())
        else:
            vec = torch.cat([source, wizard], dim = 1) if wizard is not None else source 

        coms = []
        for l in self.net:
            vec, com_tmp = l(vec)
            coms.append(com_tmp)
        
        coms = torch.cat(coms, dim = -1)
        coms = self.dropi(coms)
        

        if self.fpool is not None:
            coms = self.fpool(coms).reshape([source.shape[0], -1])
            vec = self.ff(coms)
        else:
            vec = self.dropi(vec)
            vec = torch.mean(vec, dim = 1)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls


class layer(nn.Module):
    def __init__(self, hidden_dim, percom):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sig = torch.sigmoid
        self.tan = torch.tanh
        self.relu = torch.relu
        k_size = 3

        self.conv = modules.Conv1d(hidden_dim, hidden_dim, k_size, padding = 1)
        self.line = modules.Linear(hidden_dim, hidden_dim)
        self.comp = modules.Linear(hidden_dim, percom)

    def forward(self, vec):
        vec = vec + self.relu(self.conv(vec)) + self.relu(self.line(vec))
        com = self.comp(vec)
        return vec, com






























