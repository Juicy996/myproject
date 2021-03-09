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

@register_model('shcnn')
class shcnn(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'shcnn', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.net = nn.ModuleList()
        for i in range(model_config.nlayer):
            self.net.append(block(self.hidden_dim, self.hidden_dim, 3, template.shape))


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

        # vec = vec.transpose(1, 2)
        # vec = self.conv(vec)
        # vec = vec.transpose(1, 2)
        # vec = torch.mean(vec, dim = 1)

        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls


class block(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, template_shape):
        super().__init__()
        self.conv = modules.SConv1d(in_dim, out_dim, k_size, template_shape)
        self.fnn  = modules.SLinear(in_dim, 30, template_shape)

    def forward(self, vec, template):
        vec = self.conv(vec, template)
        com = self.fnn(vec, template)
        return vec
#  -------------------------














