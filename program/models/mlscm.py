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

@register_model('mlscm')
class mlscm(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'mlscm', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.net = nn.ModuleList()
        for i in range(model_config.nlayer): 
            self.net.append(layer(self.hidden_dim))
        
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
        

        self.pool = None
        
        # self.pool = modules.FlexibleLength(self.hidden_dim, self.hidden_dim, 10)
        # self.fnn = nn.Linear(self.hidden_dim, 30)

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
        
        fstate = vec.clone()
        for l in self.net:
            fstate = l(fstate)

        vec = fstate
        
        vec = self.ff(vec)

        if self.pool:
            vec = self.pool(vec)
            vec = self.fnn(vec).reshape([-1, self.hidden_dim])
        else:
            vec = torch.mean(vec, dim = 1)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls

class layer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.sig = torch.sigmoid
        self.tanh = torch.tanh
        self.relu = torch.relu

        self.hidden_dim = hidden_dim

        self.gxfx = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.gcfx = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.gxfh = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.gcfh = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)

        self.gxdx = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.gxdh = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.gcdx = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.gcdh = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)

        self.norm = LayerNorm(hidden_dim)
        self.dropi = nn.Dropout(0.2)
        # self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, vec):
        
        # forget
        fstate = vec
        # fstate = fstate.transpose(1, 2)
        fstate = self.norm(torch.tanh(fstate))
        # fstate = fstate.transpose(1, 2)
        gxf = self.sig(self.gxfx(fstate)) * self.relu(self.gxfh(fstate))
        gcf = self.sig(self.gcfx(fstate)) * self.relu(self.gcfh(fstate))
        fstate = torch.relu(fstate * gxf * gcf) # 结合上下文和自身向量，决定应该忘记什么

        gcx = self.sig(self.gxdx(fstate)) * self.relu(self.gxfh(fstate))
        gcd = self.sig(self.gcdx(fstate)) * self.relu(self.gcfh(fstate))
        fstate = torch.relu(fstate + gcd * gcx)

        fstate = self.dropi(fstate)

        return fstate + vec































