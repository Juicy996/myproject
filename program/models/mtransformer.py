import torch
from . import basic_model, register_model
from .. import modules
import torch.nn as nn
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

@register_model('mtransformer')
class mtransformer(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mtransformer'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.net = MultiHeadAttention(self.hidden_dim, self.hidden_dim, nhead = 5)
        self.dropi = nn.Dropout(0.5)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = modules.prepare_input(batch, embedding)
        if self.transfer: vec = self.transfer(vec)
        extra_output = {'to_loss': [vec, target]}

        vec = self.net(vec, vec, vec)
        vec = self.dropi(vec)
        ret = torch.mean(vec, dim = 1)

        return ret, extra_output
    
    @classmethod
    def setup_model(cls):
        return cls

class MultiHeadAttention(nn.Module): 
    def __init__(self, 
                 in_dim,
                 out_dim,
                 nhead = 1,
                 wq = True, 
                 wk = True, 
                 wv = True, 
                 pre_ln_q = None, 
                 pre_ln_k = None, 
                 pre_ln_v = None, 
                 dropp = 0.0, 
                 initializer = nn.init.xavier_uniform_,
        ):
        super().__init__()
        assert out_dim % nhead == 0, 'hidden_dim must be divided evenly by nhead.'
        self.nhead = nhead
        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.wq = modules.Linear(in_dim, out_dim, initializer = initializer) if wq else None
        self.wk = modules.Linear(in_dim, out_dim, initializer = initializer) if wk else None
        self.wv = modules.Linear(in_dim, out_dim, initializer = initializer) if wv else None
        self.pre_ln_q = pre_ln_q
        self.pre_ln_k = pre_ln_k
        self.pre_ln_v = pre_ln_v

    def forward(self, query, key, value, attn_mask = None):
        if self.pre_ln_q: query = self.pre_ln_q(query)
        if self.pre_ln_k: key   = self.pre_ln_k(key)
        if self.pre_ln_v: value = self.pre_ln_v(value)

        if self.wq is not None: query = self.wq(query)
        if self.wk is not None: key   = self.wk(key)
        if self.wv is not None: value = self.wv(value)
        
        ret = modules.attention_dir(query, key, value, self.nhead, torch.relu, self.dropi, attn_mask, batch_first = True)

        if self.dropi:    ret = self.dropi(ret)

        return ret
        




















