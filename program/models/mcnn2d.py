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

@register_model('mcnn2d')
class mcnn2d(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'indicator', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.net = nn.ModuleList([layer(self.hidden_dim, self.hidden_dim, 3) for i in range(model_config.nlayer)])
        self.wiz = commander1(self.net, self.hidden_dim)

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

        # selected = self.wiz(vec, self.net)
        # print(selected)
        for layer in self.net:
        # for layer, cmd in zip(self.net, selected):
            cmd = self.wiz(vec, layer)
            res = vec
            # if cmd == 0: continue
            vec = torch.einsum('bik,b->bik', layer(vec), cmd) + res
        vec = torch.mean(vec, dim = 1)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls


@register_model('mcnn1d')
class mcnn1d(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'mcnn1d', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.net = nn.ModuleList([Conv1d(self.hidden_dim, self.hidden_dim, 3) for i in range(model_config.nlayer)])

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

        for layer in self.net:
            vec = layer(vec)

        vec = torch.mean(vec, dim = 1)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls

class Conv1d(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, stride = 1, padding = 0, initializer = nn.init.xavier_uniform_):
        super().__init__()
        self.net = nn.Conv1d(in_dim, out_dim, k_size, stride = stride, padding = padding)
        initializer(self.net.weight)
        nn.init.constant_(self.net.bias, 1.0)

    def forward(self, vec):
        vec = vec.transpose(1, 2)
        vec = self.net(vec)
        vec = vec.transpose(1, 2)
        return vec





















