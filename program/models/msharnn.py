import torch, math, random, torch.utils, torch.utils.checkpoint
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import basic_model, register_model
from .. import modules

if True:
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    except:
        from torch.nn import LayerNorm
else:
    from torch.nn import LayerNorm

class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, max_mem, nhead = 1, dropp = 0.1, residual = True, rnn = True, use_attn = True):
        super().__init__()
        self.attn = None
        if use_attn: self.attn = modules.EMultiHeadAttention(embed_dim, hidden_dim, nhead = nhead, dropp = dropp, residual = residual)
        if rnn: self.rnn = LSTM(embed_dim, embed_dim, batch_first = False)
        self.ff = modules.Boom(embed_dim, hidden_dim, dropp = dropp, shortcut = True, residual = residual)

        self.max_mem = max_mem
        self.lnmid = LayerNorm(embed_dim, eps=1e-12)
        self.lnmem = LayerNorm(embed_dim, eps=1e-12)
        self.lnout = LayerNorm(embed_dim, eps=1e-12)
        self.lnff = LayerNorm(embed_dim, eps=1e-12)
        self.gelu = modules.GELU()   

    def forward(self, vec, attn_mask = None, mem = None, hidden = None):
        
        if self.rnn: ret, new_hidden = self.rnn(vec, None if hidden is None else hidden)
        
        new_mem = []
        if self.attn is not None:
            mh = self.lnmem(ret)
            h  = self.lnmid(ret)
            bigh = torch.cat([mem, mh], dim = 0) if mem is not None else mh
            new_mem = bigh[-self.max_mem:]

            q, k = ret, bigh

            ret = self.attn(q, k, bigh, attn_mask = attn_mask)

        if self.ff: ret = self.ff(ret)
        return ret, new_mem, new_hidden

@register_model('msharnn')
class SHARNN(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        self.nlayer = model_config.nlayer
        self.nhead = model_config.nhead
        self.idrop = nn.Dropout(model_config.dropp)
        self.hdrop = nn.Dropout(model_config.dropp)

        self.blocks = nn.ModuleList()
        for idx in range(self.nlayer):
            uta = True if idx == self.nlayer - 2 else False
            self.blocks.append(
                Block(self.embed_dim, self.hidden_dim, 5000, self.nhead, dropp = model_config.dropp, residual = True, use_attn = uta)
            )

    def forward(self, batch, embedding, extra_input, template, writer):
        """ Input has shape [seq length, batch] """
        ret = {'mems': [], 'hids': []}
        # source, wizard, target = batch
        vec = modules.prepare_input(batch, embedding)
        vec = vec.transpose(0, 1)
        vec = self.idrop(vec)

        mems = None if extra_input is None else extra_input['mems']
        hidden = None if extra_input is None else extra_input['hidden']

        new_hidden = []
        new_mems = []

        attn_mask = torch.full((len(vec), len(vec)), -float('Inf'), device = vec.device, dtype = vec.dtype)
        attn_mask = torch.triu(attn_mask, diagonal = 1)

        if mems:
            max_mems = max(len(m) for m in mems)
            happy = torch.zeros((len(vec), max_mems), device = vec.device, dtype = vec.dtype)
            attn_mask = torch.cat([happy, attn_mask], dim = -1)

        for idx, block in enumerate(self.blocks):
            mem = mems[idx] if mems else None
            hid = hidden[idx] if hidden else None
            vec, m, nh = block(vec, attn_mask = attn_mask, mem = mem, hidden = hid)
            new_hidden.append(nh)
            new_mems.append(m)

        rep = self.hdrop(vec).transpose(0, 1)
        ret['mems'] = new_mems
        ret['hidden'] = new_hidden

        return rep, ret

    @classmethod
    def setup_model(cls):
        return cls

class LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, batch_first = False):
        super().__init__()
        self.net = nn.LSTM(input_size = in_dim, hidden_size = out_dim, batch_first = batch_first)

    def forward(self, vec, hidden):
        self.net.flatten_parameters()
        vec, new_hidden = self.net(vec, hidden)
        return vec, new_hidden
   





































