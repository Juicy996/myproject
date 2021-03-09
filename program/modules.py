import os, sys, time, random, torch, math, warnings
import numpy as np
import torch.nn as nn
from . import utils 
from transformers import AutoTokenizer, AutoModel
from program.datasets import datasets_utils
if True:
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    except:
        warnings.warn("apex.normalization.fused_layer_norm is unavailable, using nn.LayerNorm instead.")
        from torch.nn import LayerNorm
else:
    from torch.nn import LayerNorm

def attention_for(query, key, value, nhead, activate, dropi, attn_mask, batch_first):
    hidden_dim = query.shape[-1]
    query = torch.split(query, hidden_dim // nhead, dim = -1)
    key   = torch.split(key  , hidden_dim // nhead, dim = -1)
    value = torch.split(value, hidden_dim // nhead, dim = -1)
    h_dim = hidden_dim // nhead
    ret = []
    if batch_first:
        for q, k, v in zip(query, key, value):
            attn_score = torch.einsum('bid,bjd->bij', q, k).contiguous() / h_dim
            if attn_mask is not None: 
                attn_score = attn_score + attn_mask if -float('Inf') in attn_mask else attn_score * attn_mask
            if activate: attn_score = activate(attn_score)
            attn_score = torch.nn.functional.softmax(attn_score, dim = -1)  # 为什么是-1
            if dropi: attn_score = dropi(attn_score)
            ret_tmp = torch.einsum('bij,bjd->bid', attn_score, v)
            ret.append(ret_tmp)
    else:
        for q, k, v in zip(query, key, value):
            attn_score = torch.einsum('ibd,jbd->bij', q, k).contiguous() / h_dim
            if attn_mask is not None: 
                attn_score = attn_score + attn_mask if -float('Inf') in attn_mask else attn_score * attn_mask
            if activate: attn_score = activate(attn_score)
            attn_score = torch.nn.functional.softmax(attn_score, dim = -1)
            if dropi: attn_score = dropi(attn_score)
            ret_tmp = torch.einsum('bij,jbd->ibd', attn_score, v)
            ret.append(ret_tmp)
    ret = torch.cat(ret, dim = -1)
    return ret

def attention_dir(query, key, value, nhead, activate, dropi, attn_mask, batch_first, printmat = False):
    hidden_dim = query.shape[-1]
    h_dim = hidden_dim // nhead
    scale = h_dim ** 0.5

    query = query.reshape([*query.shape[:2], nhead, h_dim])
    key   = key.reshape([*key.shape[:2], nhead, h_dim])
    value = value.reshape([*value.shape[:2], nhead, h_dim])
    
    if batch_first:
        attn_score = torch.einsum('bihd,bjhd->bhij', query, key).contiguous() / scale
        if attn_mask is not None: 
            attn_score = attn_score + attn_mask if -float('Inf') in attn_mask else attn_score * attn_mask
        if activate: attn_score = activate(attn_score)
        # attn_score = torch.nn.functional.softmax(attn_score, dim = 1)  # 为什么是-1
        if printmat:
            utils.heatmap(attn_score[0][0].cpu().detach())
        if dropi: attn_score = dropi(attn_score)
        ret = torch.einsum('bhij,bjhd->bihd', attn_score, value).reshape([*query.shape[:2], hidden_dim])
    else:
        attn_score = torch.einsum('ibhd,jbhd->bhij', query, key).contiguous() / scale
        if attn_mask is not None: 
            attn_score = attn_score + attn_mask if -float('Inf') in attn_mask else attn_score * attn_mask
        if activate: attn_score = activate(attn_score)
        attn_score = torch.nn.functional.softmax(attn_score, dim = -1)
        if dropi: attn_score = dropi(attn_score)
        ret = torch.einsum('bhij,jbhd->ibhd', attn_score, value).reshape([*query.shape[:2], hidden_dim])
    return ret

def prepare_input(batch, embedding):
    source, wizard, target = batch

    if embedding is not None:
        max_slen = torch.max(torch.sum(torch.ne(source, 0).float(), dim = 1)).long()
        source = source[:, :max_slen]

        if wizard is not None:
            max_wlen = torch.max(torch.sum(torch.ne(wizard, 0).float(), dim = 1)).long()
            wizard = wizard[:, :max_slen]
        if isinstance(embedding, nn.ModuleDict):
            vec = embedding['emb_src'](source.long())
            wiz = embedding['emb_tgt'](wizard.long())
            vec = torch.cat([wiz, vec], dim = 1)
        else:
            vec = embedding(source.long())
    else:
        if wizard is not None:
            vec = torch.cat([source, wizard], dim = 1)
        else:
            vec = source
    return vec

class Noise(nn.Module):
    def __init__(self, scale = 0.2):
        super().__init__()
        self.scale = scale
    def forward(self, vec):
        noise = torch.normal(0, vec * self.scale, out = None)
        ret = vec + noise
        return ret

class Embedding(nn.Module):
    def __init__(self, 
                 ntoken, 
                 embed_dim, 
                 pretrained_matrix = None, 
                 trainable = True, 
                 dropout = None,
                 initializer = nn.init.xavier_uniform_,
                ):
        super().__init__()
        if pretrained_matrix is None:
            self.embedding = nn.Embedding(ntoken, embed_dim)
            initializer(self.embedding.weight)
            
        elif isinstance(pretrained_matrix, np.ndarray):
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_matrix), freeze = not trainable)
        
        elif pretrained_matrix in ['bert-base-uncased']:
            self.embedding = datasets_utils.transformer_model(pretrained_matrix)
            utils.set_freeze_all(self.embedding, freeze = not trainable)
        else:
            raise ValueError(f'Unknown pretrained style [{pretrained_matrix}].')
        
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
    
    def forward(self, vec):
        ret = self.embedding(vec)
        if self.dropout: ret = self.dropout(ret)
        return ret

class PosEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad = False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.left_pad = left_pad
    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = Variable(input.data.new(1, 1).fill_(self.padding_idx + input.size(1)))
        else:
            positions = Variable(self.make_positions(input.data))
        return embedding(positions)
    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1
    def make_positions(self, input):
        """Replace non-padding symbols with their position numbers."""
        if not hasattr(self, 'range_buf'):
            self.range_buf = input.new()
        seqlen = input.size(1)
        if self.range_buf.numel() < seqlen:
            # offset positions by the padding index
            torch.arange(self.padding_idx + 1, self.padding_idx + 1 + seqlen, out=self.range_buf)
        mask = input.ne(self.padding_idx)
        positions = self.range_buf[:seqlen].expand_as(input)
        if self.left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
        return input.clone().masked_scatter_(mask, positions[mask])

class PositionalEmbedding(nn.Module): # seq_len. batch, dim, transxl
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

# normal modules
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, initializer = nn.init.xavier_uniform_):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        initializer(self.net.weight)
        nn.init.constant_(self.net.bias, 1.0)
        
    def forward(self, vec):
        vec = self.net(vec)
        return vec

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

class LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, batch_first = False):
        super().__init__()
        self.net = nn.LSTM(input_size = in_dim, hidden_size = out_dim, batch_first = batch_first)

    def forward(self, vec, hidden):
        self.net.flatten_parameters()
        vec, new_hidden = self.net(vec, hidden)
        return vec, new_hidden

class MultiHeadAttention(nn.Module): 
    def __init__(self, 
                 in_dim,
                 out_dim,
                 nhead = 1,
                 wq = True, 
                 wk = True, 
                 wv = True, 
                 batch_first = False,
                 dropp = 0.0, 
                 activate_q = None,
                 activate_k = None, 
                 activate_v = None,
                 activate_a = None,
                 activate_r = None, 
                 pre_ln_q = None, 
                 pre_ln_k = None, 
                 pre_ln_v = None, 
                 post_ln = None, 
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
        ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        assert out_dim % nhead == 0, 'hidden_dim must be divided evenly by nhead.'
        if (pre_ln_q or pre_ln_k or pre_ln_v) and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.pre_ln_q = pre_ln_q
        self.pre_ln_k = pre_ln_k
        self.pre_ln_v = pre_ln_v
        self.activate_q = activate_q
        self.activate_k = activate_k
        self.activate_v = activate_v
        self.activate_a = activate_a
        self.activate_r = activate_r
        self.post_ln  = post_ln
        self.residual = residual
        
        self.wq = Linear(in_dim, out_dim, initializer = initializer) if wq else None
        self.wk = Linear(in_dim, out_dim, initializer = initializer) if wk else None
        self.wv = Linear(in_dim, out_dim, initializer = initializer) if wv else None

        self.batch_first = batch_first
        self.hidden_dim = out_dim
        self.nhead = nhead

    def forward(self, query, key, value, target_pos = None, attn_mask = None):
        residual = query
        if self.pre_ln_q: query = self.pre_ln_q(query)
        if self.pre_ln_k: key   = self.pre_ln_k(key)
        if self.pre_ln_v: value = self.pre_ln_v(value)

        if self.wq is not None: query = self.wq(query)
        if self.wk is not None: key   = self.wk(key)
        if self.wv is not None: value = self.wv(value)

        if self.activate_q: query = self.activate_q(query)
        if self.activate_k: key   = self.activate_k(key)
        if self.activate_v: value = self.activate_v(value)
        
        ret = attention_dir(query, key, value, self.nhead, self.activate_a, self.dropi, attn_mask, self.batch_first)

        if self.activate_r: ret = self.activate_r(ret)
        if self.post_ln:  ret = self.post_ln(ret)
        if self.dropi:    ret = self.dropi(ret)
        if self.residual: ret = ret + residual

        return ret

class EMultiHeadAttention(nn.Module): 
    def __init__(self, 
                 in_dim,
                 out_dim,
                 nhead = 1,
                 wq = True, 
                 wk = True, 
                 wv = True,
                 batch_first = False,
                 dropp = 0.0, 
                 activate_q = None,
                 activate_k = None, 
                 activate_v = None,
                 activate_a = None,
                 activate_r = None,
                 pre_ln_q = None, 
                 pre_ln_k = None, 
                 pre_ln_v = None, 
                 post_ln = None, 
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
        ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        assert out_dim % nhead == 0, 'hidden_dim must be divided evenly by nhead.'
        if (pre_ln_q or pre_ln_k or pre_ln_v) and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.pre_ln_q = pre_ln_q
        self.pre_ln_k = pre_ln_k
        self.pre_ln_v = pre_ln_v
        self.activate_q = activate_q
        self.activate_k = activate_k
        self.activate_v = activate_v
        self.activate_a = activate_a
        self.activate_r = activate_r
        self.post_ln  = post_ln
        self.residual = residual

        self.qs = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.ks = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.vs = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.qkvs = nn.Parameter(torch.FloatTensor(1, 3, out_dim))
        initializer(self.qs)
        initializer(self.ks)
        initializer(self.vs)
        initializer(self.qkvs)
        self.gelu = GELU()
        
        self.wq = Linear(in_dim, out_dim, initializer = initializer) if wq else None
        self.wk = Linear(in_dim, out_dim, initializer = initializer) if wk else None
        self.wv = Linear(in_dim, out_dim, initializer = initializer) if wv else None
        self.r_gate = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.vq = Overparam(out_dim)

        self.batch_first = batch_first
        self.hidden_dim = out_dim
        self.nhead = nhead

    def forward(self, query, key, value, target_pos = None, attn_mask = None):
        residual = query
        if self.pre_ln_q: query = self.pre_ln_q(query)
        if self.pre_ln_k: key = self.pre_ln_k(key)
        if self.pre_ln_v: value = self.pre_ln_v(value)

        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), self.vq(torch.sigmoid(self.vs))

        if self.wq is not None: query = self.wq(query)
        if self.wk is not None: key   = self.wk(key)
        if self.wv is not None: value = self.wv(value)
        query, key, value = query + qs, key + ks, value + vs

        if self. activate_q: query = self.activate_q(query)
        if self. activate_k: key   = self.activate_k(key)
        if self. activate_v: value = self.activate_v(value)
        
        ret = attention_dir(query, key, value, self.nhead, self.activate_a, self.dropi, attn_mask, self.batch_first)

        if self.activate_r: ret = self.activate_r(ret)
        if self.post_ln:  ret = self.post_ln(ret)
        if self.dropi:    ret = self.dropi(ret)
        if self.residual: ret = ret + residual

        return ret

class CTM(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 k_size, 
                 dropp = 0.0, 
                 activate = None, 
                 pre_ln = None, 
                 post_ln = None, 
                 residual = False,
                ):
        super().__init__() # T B C
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        if pre_ln and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate = activate
        self.pre_ln = None if pre_ln is None else pre_ln(in_dim)
        self.post_ln = None if post_ln is None else post_ln(out_dim)
        self.residual = residual

        self.k_size = k_size

        self.conv = ConvTBC(in_dim, out_dim, k_size = k_size, padding = 'valid')
        self.linear = Linear(in_dim, out_dim)
        self.olin = Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def forward(self, vec, hidden):
        residual = vec
        if self.pre_ln: vec = self.pre_ln(vec)
        
        if hidden is not None:
            hidden_state, cell_state = hidden
        else:
            hidden_state = torch.randn([self.k_size - 1, self.out_dim])
            cell_state   = torch.randn([self.k_size - 1, self.out_dim])
        vec = torch.cat([hidden_state.unsqueeze(1).repeat(1, vec.shape[1], 1), vec], dim = 0)
        
        ret = []
        new_h = []
        new_c = cell_state
        for idx in range(len(vec) - self.k_size + 1):
            lv = vec[idx + self.k_size - 1]
            cv = self.conv(vec[idx: idx + self.k_size])
            h_tmp = new_c * lv + cv
            new_c = h_tmp * cv
            new_h.append(h_tmp)
            ret.append(self.olin(h_tmp))
        new_h = torch.cat(new_h, dim = 0)
        ret = torch.cat(ret, dim = 0)

        if self.activate: ret = self.activate(ret)
        if self.post_ln: ret = self.post_ln(ret)
        if self.dropi: ret = self.dropi(ret)
        if self.residual: ret = ret + residual

        return ret, (new_h, new_c)

class Overparam(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 dropp = 0.0,
                 activate = None,
                 pre_ln = None, 
                 post_ln = None, 
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
                ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        if pre_ln and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')
        self.l1 = Linear(hidden_dim, 2 * hidden_dim, initializer = initializer)
        self.hidden_dim = hidden_dim

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate = activate
        self.pre_ln = pre_ln
        self.post_ln = post_ln
        self.residual = residual

    def forward(self, x):
        if self.pre_ln: x = self.pre_ln(x)
        c, f = self.l1(x).split(self.hidden_dim, dim = -1)
        ret = torch.sigmoid(f) * torch.tanh(c)

        if self.activate: ret = self.activate(ret)
        if self.post_ln: ret = self.post_ln(ret)
        if self.dropi: ret = self.dropi(ret)
        if self.residual: ret = ret + residual
        return ret

class Boom(nn.Module):
    def __init__(self, 
                 i_dim, 
                 inter = 0, 
                 shortcut = False, 
                 dropp = 0.0, 
                 activate = None, 
                 pre_ln = None, 
                 post_ln = None, 
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
                ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        if pre_ln and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate = activate
        self.pre_ln = pre_ln
        self.post_ln = post_ln
        self.residual = residual

        if inter == 0: inter = 4 * i_dim

        self.linear1 = Linear(i_dim, inter, initializer = initializer)
        self.dropi = nn.Dropout(dropp) if dropp > 0 else None
        if not shortcut:
            self.linear2 = Linear(inter, i_dim, initializer = initializer)
        self.shortcut = shortcut
        self.act = GELU()

    def forward(self, vec):
        residual = vec
        if self.pre_ln: vec = self.pre_ln(vec)

        ret = self.linear1(vec)
        ret = self.act(ret)
        if self.dropi: ret = self.dropi(ret)

        if self.shortcut:
            ninp = vec.shape[-1]
            ret = torch.narrow(ret, -1, 0, ret.shape[-1] // ninp * ninp)
            ret = ret.view(*ret.shape[:-1], ret.shape[-1] // ninp, ninp)
            ret = ret.sum(dim = -2)
        else:
            ret = self.linear2(ret)
        
        if self.activate: ret = self.activate(ret)
        if self.post_ln: ret = self.post_ln(ret)
        if self.dropi: ret = self.dropi(ret)
        if self.residual: ret = ret + residual
        
        return ret

# shared modules
class SLinear(nn.Module):
    def __init__(self, in_dim, out_dim, template_shape, bias = True, initializer = nn.init.xavier_uniform_):
        super().__init__()
        self.conv = nn.Conv2d(template_shape[1], 1, (template_shape[3] - out_dim + 1, template_shape[2] - in_dim + 1))
        initializer(self.conv.weight)
        if bias:
            bias = torch.FloatTensor(1, out_dim)
            initializer(bias)
            self.bias = nn.Parameter(bias.squeeze())
        else:
            self.bias = 0

    def forward(self, vec, template):
        weight = self.conv(template).squeeze()
        ret = torch.nn.functional.linear(vec, weight) + self.bias
        return ret

class SConv1d(nn.Module):# input_shape = [TBC]
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 k_size,
                 template_shape, 
                 stride = 1,
                 padding = 0,
                 initializer = nn.init.xavier_uniform_,
                 bias = True,
        ):
        super().__init__()

        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2d(template_shape[1], k_size, (template_shape[2] - in_dim + 1, template_shape[3] - out_dim + 1))
        initializer(self.conv.weight)
        if bias:
            bias = initializer(torch.FloatTensor(1, out_dim))
            self.bias = nn.Parameter(bias.squeeze()) 
        else:
            self.bias = None
    
    def forward(self, vec, template):
        weight = self.conv(template).squeeze().permute(2, 1, 0)
        vec = vec.transpose(1, 2)
        vec = torch.nn.functional.conv1d(vec, weight, self.bias, stride = self.stride, padding = self.padding)
        vec = vec.transpose(1, 2)
        return vec

class SMultiHeadAttention(nn.Module): 
    def __init__(self,
                 in_dim,
                 out_dim,
                 template_shape,
                 nhead = 1,
                 wq = True,
                 wk = True,
                 wv = True,
                 batch_first = False,
                 k_size = 3,
                 mode = 'linear',
                 dropp = 0.0,
                 activate_q = None,
                 activate_k = None,
                 activate_v = None,
                 activate_a = None,
                 activate_r = None,
                 pre_ln_q = None,
                 pre_ln_k = None,
                 pre_ln_v = None,
                 post_ln = None,
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
        ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        assert out_dim % nhead == 0, 'hidden_dim must be divided evenly by nhead.'
        if (pre_ln_q or pre_ln_k or pre_ln_v) and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate_q = activate_q
        self.activate_k = activate_k
        self.activate_v = activate_v
        self.activate_a = activate_a
        self.activate_r = activate_r
        self.pre_ln_q = pre_ln_q
        self.pre_ln_k = pre_ln_k
        self.pre_ln_v = pre_ln_v
        self.post_ln  = post_ln
        self.residual = residual
        
        if mode == 'linear':
            self.wq = SLinear(in_dim, out_dim, template_shape, initializer = initializer) if wq else None
            self.wk = SLinear(in_dim, out_dim, template_shape, initializer = initializer) if wk else None
            self.wv = SLinear(in_dim, out_dim, template_shape, initializer = initializer) if wv else None
        elif mode == 'conv':
            self.wq = SConv1d(in_dim, out_dim, k_size, template_shape, padding = 'mask', initializer = initializer) if wq else None
            self.wk = SConv1d(in_dim, out_dim, k_size, template_shape, padding = 'mask', initializer = initializer) if wk else None
            self.wv = SConv1d(in_dim, out_dim, k_size, template_shape, padding = 'mask', initializer = initializer) if wv else None
        else:
            raise ValueError('Unknown mode [conv, linear]')
        self.batch_first = batch_first
        self.hidden_dim = out_dim
        self.nhead = nhead

    def forward(self, query, key, value, template, target_pos = None, attn_mask = None):
        residual = query
        if self.pre_ln_q: query = self.pre_ln_q(query)
        if self.pre_ln_k: key = self.pre_ln_k(key)
        if self.pre_ln_v: value = self.pre_ln_v(value)

        if self.wq is not None: query = self.wq(query, template)
        if self.wk is not None: key   = self.wk(key  , template)
        if self.wv is not None: value = self.wv(value, template)

        if self. activate_q: query = self.activate_q(query)
        if self. activate_k: key   = self.activate_k(key)
        if self. activate_v: value = self.activate_v(value)

        ret = attention_dir(query, key, value, self.nhead, self.activate_a, self.dropi, attn_mask, self.batch_first)
        
        if self.activate_r: ret = self.activate_r(ret)
        if self.post_ln:  ret = self.post_ln(ret)
        if self.dropi:    ret = self.dropi(ret)
        if self.residual: ret = ret + residual

        return ret

class SEMultiHeadAttention(nn.Module): 
    def __init__(self, 
                 in_dim,
                 out_dim,
                 template_shape,
                 nhead = 1,
                 wq = True, 
                 wk = True, 
                 wv = True,
                 batch_first = False,
                 dropp = 0.0, 
                 activate_q = None,
                 activate_k = None,
                 activate_v = None,
                 activate_a = None,
                 activate_r = None,
                 pre_ln_q = None, 
                 pre_ln_k = None, 
                 pre_ln_v = None, 
                 post_ln = None, 
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
        ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        assert out_dim % nhead == 0, 'hidden_dim must be divided evenly by nhead.'
        if (pre_ln_q or pre_ln_k or pre_ln_v) and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate_q = activate_q
        self.activate_k = activate_k
        self.activate_v = activate_v
        self.activate_a = activate_a
        self.activate_r = activate_r
        self.pre_ln_q = pre_ln_q
        self.pre_ln_k = pre_ln_k
        self.pre_ln_v = pre_ln_v
        self.post_ln  = post_ln
        self.residual = residual

        self.qs = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.ks = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.vs = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.qkvs = nn.Parameter(torch.FloatTensor(1, 3, out_dim))
        initializer(self.qs)
        initializer(self.ks)
        initializer(self.vs)
        initializer(self.qkvs)
        self.gelu = GELU()
        
        self.wq = SLinear(in_dim, out_dim, template_shape, initializer = initializer) if wq else None
        self.wk = SLinear(in_dim, out_dim, template_shape, initializer = initializer) if wk else None
        self.wv = SLinear(in_dim, out_dim, template_shape, initializer = initializer) if wv else None
        self.r_gate = nn.Parameter(torch.FloatTensor(1, 1, out_dim))
        self.vq = Overparam(out_dim)

        self.batch_first = batch_first
        self.hidden_dim = out_dim
        self.nhead = nhead

    def forward(self, query, key, value, template, target_pos = None, attn_mask = None):
        residual = query
        if self.pre_ln_q: query = self.pre_ln_q(query)
        if self.pre_ln_k: key = self.pre_ln_k(key)
        if self.pre_ln_v: value = self.pre_ln_v(value)

        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), self.vq(torch.sigmoid(self.vs))

        if self.wq is not None: query = self.wq(query, template)
        if self.wk is not None: key   = self.wk(key, template)
        if self.wv is not None: value = self.wv(value, template)
        query, key, value = query + qs, key + ks, value + vs

        if self. activate_q: query = self.activate_q(query)
        if self. activate_k: key   = self.activate_k(key)
        if self. activate_v: value = self.activate_v(value)
        
        ret = attention_dir(query, key, value, self.nhead, self.activate_a, self.dropi, attn_mask, self.batch_first)

        if self.activate_r: ret = self.activate_r(ret)
        if self.post_ln:  ret = self.post_ln(ret)
        if self.dropi:    ret = self.dropi(ret)
        if self.residual: ret = ret + residual

        return ret

class SCTM(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, template_shape, dropp = 0.0, activate = None, pre_ln = None, post_ln = None, residual = False):
        super().__init__() # T B C
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        if pre_ln and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate = activate
        self.pre_ln = None if pre_ln is None else pre_ln(in_dim)
        self.post_ln = None if post_ln is None else post_ln(out_dim)
        self.residual = residual

        self.k_size = k_size

        self.conv    = SConvTBC(in_dim, out_dim, k_size, template_shape, padding = 'valid')
        self.linear  = SLinear(in_dim, out_dim, template_shape)
        self.olin    = SLinear(in_dim, out_dim, template_shape)
        self.in_dim  = in_dim
        self.out_dim = out_dim
        
    def forward(self, vec, template, hidden):
        residual = vec
        if self.pre_ln: vec = self.pre_ln(vec)
        
        if hidden is not None:
            hidden_state, cell_state = hidden
        else:
            hidden_state = torch.randn([self.k_size - 1, self.out_dim])
            cell_state   = torch.randn([self.k_size - 1, self.out_dim])
        vec = torch.cat([hidden_state.unsqueeze(1).repeat(1, vec.shape[1], 1), vec], dim = 0)
        
        ret = []
        new_h = []
        new_c = cell_state
        for idx in range(len(vec) - self.k_size + 1):
            lv = self.linear(vec[idx + self.k_size - 1], template)
            cv = self.conv(vec[idx: idx + self.k_size], template)
            h_tmp = new_c * lv + cv
            new_c = h_tmp * cv
            new_h.append(h_tmp)
            ret.append(self.olin(h_tmp, template))
        new_h = torch.cat(new_h, dim = 0)
        ret = torch.cat(ret, dim = 0)

        if self.activate: ret = self.activate(ret)
        if self.post_ln: ret = self.post_ln(ret)
        if self.dropi: ret = self.dropi(ret)
        if self.residual: ret = ret + residual

        return ret, (new_h, new_c)

class SOverparam(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 template_shape,
                 dropp = 0.0,
                 activate = None,
                 pre_ln = None, 
                 post_ln = None, 
                 residual = False,
                 initializer = nn.init.xavier_uniform_,
                ):
        super().__init__()
        assert isinstance(dropp, float), 'Type Error of dropp'
        assert residual in [True, False], 'Type Error of residual'
        if pre_ln and post_ln: warnings.warn('Pre_ln and post_ln will be executed in the same module.')
        self.l1 = SLinear(hidden_dim, 2 * hidden_dim, template_shape, initializer = initializer)
        self.hidden_dim = hidden_dim

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)
        self.activate = activate
        self.pre_ln = pre_ln
        self.post_ln = post_ln
        self.residual = residual

    def forward(self, x, template):
        if self.pre_ln: x = self.pre_ln(x)
        c, f = self.l1(x, template).split(self.hidden_dim, dim = -1)
        ret = torch.sigmoid(f) * torch.tanh(c)

        if self.activate: ret = self.activate(ret)
        if self.post_ln: ret = self.post_ln(ret)
        if self.dropi: ret = self.dropi(ret)
        if self.residual: ret = ret + residual
        return ret

class SBoom(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 template_shape,
                 inter = 0, 
                 shortcut = False, 
                 initializer = nn.init.xavier_uniform_,
                ):
        super().__init__()
        self.linear1 = SLinear(hidden_dim, hidden_dim, template_shape, initializer = initializer)
        self.dropi = nn.Dropout(dropp) if dropp > 0 else None
        if not shortcut:
            self.linear2 = SLinear(hidden_dim, hidden_dim, template_shape, initializer = initializer)
        self.shortcut = shortcut
        self.act = GELU()

    def forward(self, vec, template):
        ret = self.linear1(vec, template)
        ret = self.act(ret)
        if self.dropi: ret = self.dropi(ret)

        if self.shortcut:
            ninp = vec.shape[-1]
            ret = torch.narrow(ret, -1, 0, ret.shape[-1] // ninp * ninp)
            ret = ret.view(*ret.shape[:-1], ret.shape[-1] // ninp, ninp)
            ret = ret.sum(dim = -2)
        else:
            ret = self.linear2(ret, template)

        return ret

# hybrid
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class GELU(nn.Module):
    def forward(self, x):
        #return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)

class FlexiblePool(nn.Module):
    def __init__(self, out_len, bins, mode = 'max', ceil_mode = True, hidden_dim = 0):
        super().__init__()
        if mode == 'max': self.pool = torch.nn.functional.max_pool1d
        elif mode == 'avg': self.pool = torch.nn.functional.avg_pool1d
        else: raise ValueError(f'Unknown pool mode [{mode}].')
        self.ceil_mode = ceil_mode
        self.out_len = out_len
        self.bins = bins
        assert sum(bins) == out_len, f'{sum(bins)} must equal to {out_len}.'
    
    def forward(self, vec, lens = None, dim_last = True):
        if dim_last: vec = vec.transpose(1, 2)
        
        ret = []
        for idx in range(len(self.bins)):
            k_size = int (math.ceil(vec.shape[-1] / float(self.bins[idx]))) 
            k_step = int(math.ceil(vec.shape[-1] / float(self.bins[idx])))

            tmp = self.pool(vec, k_size, k_step, ceil_mode = self.ceil_mode)
            ret.append(tmp)

        ret = torch.cat(ret, dim = -1)
        if dim_last: ret = ret.transpose(1, 2)
        assert ret.shape[1] == self.out_len, f'ret [shape = {ret.shape}] must = output len [{self.out_len}].'
        return ret
        
class FlexiblePatt(nn.Module):
    def __init__(self, out_len, bins, hidden_dim):
        super().__init__()
        self.wq = Linear(hidden_dim, hidden_dim)
        self.wk = Linear(hidden_dim, hidden_dim)
        self.wv = Linear(hidden_dim, hidden_dim)

        self.out_len = out_len
        self.bins = bins

        self.pools = nn.ModuleList()
        for item in bins: self.pools.append(torch.nn.AdaptiveAvgPool1d(item))
        assert sum(bins) == out_len, f'{sum(bins)} must equal to {out_len}.'
    def forward(self, vec, lens = None, dim_last = True):
        q = self.wq(vec)
        k = self.wk(vec)
        attn = torch.einsum('bik,bjk->bij', q, k)
        attn = torch.tanh(attn)
        attn = torch.softmax(attn, dim = -1)

        ret = []
        for idx, ol in enumerate(self.pools):
            tmp = ol(attn)
            ret.append(tmp)

        attn = torch.cat(ret, dim = -1)
        ret = torch.einsum('bji,bjk->bik', attn, self.wv(vec))
        assert ret.shape[1] == self.out_len, f'ret [shape = {ret.shape}] must = output len [{self.out_len}].'
        return ret

class FlexibleLQmul(nn.Module):
    def __init__(self, out_len, hidden_dim, bias = False):
        super().__init__()
        assert hidden_dim % out_len == 0, 'must.'
        hdim = hidden_dim // out_len
        self.wq = Linear(hidden_dim, out_len)
        self.wk = Linear(hidden_dim, hidden_dim)
        self.wv = Linear(hidden_dim, hdim)
        self.bias = nn.Parameter(torch.FloatTensor(out_len, hdim))if bias else None

    def forward(self, vec):
        attn = torch.einsum('bji, bjk->bij', self.wq(vec), self.wk(vec))
        attn = torch.tanh(attn)
        attn = torch.softmax(attn, dim = -1)
        ret = torch.einsum('bij,bjk->bik', attn, self.wv(vec))
        if self.bias is not None: ret = ret + self.bias
        return ret

class FlexibleLKmul(nn.Module):
    def __init__(self, out_len, hidden_dim, bias = True):
        super().__init__()
        self.wq = Linear(hidden_dim, hidden_dim)
        self.wk = Linear(hidden_dim, out_len)
        self.wv = Linear(hidden_dim, hidden_dim)
        self.bias = nn.Parameter(torch.FloatTensor(out_len, hidden_dim))if bias else None

    def forward(self, vec):
        attn = torch.einsum('bjk, bji->bij', self.wq(vec), self.wk(vec))
        attn = torch.tanh(attn)
        attn = torch.softmax(attn, dim = -1)
        ret = torch.einsum('bij,bjk->bik', attn, self.wv(vec))
        if self.bias is not None: ret = ret + self.bias
        return ret

class FlexibleQmul(nn.Module):
    def __init__(self, out_len, hidden_dim):
        super().__init__()
        self.wq = nn.Parameter(torch.FloatTensor(out_len, hidden_dim))
        self.wk = Linear(hidden_dim, hidden_dim)
        self.wv = Linear(hidden_dim, hidden_dim)
    def forward(self, vec):
        attn = torch.einsum('ik, bjk->bij', self.wq, self.wk(vec))
        ret = torch.einsum('bij,bjk->bik', attn, self.wv(vec))
        return ret

class FlexibleKmul(nn.Module):
    def __init__(self, out_len, hidden_dim):
        super().__init__()
        self.wq = Linear(hidden_dim, hidden_dim)
        self.wk = nn.Parameter(torch.FloatTensor(out_len, hidden_dim))
        self.wv = Linear(hidden_dim, hidden_dim)
    def forward(self, vec):
        attn = torch.einsum('bik, jk->bij', self.wq(vec), self.wk)
        ret = torch.einsum('bji,bjk->bik', attn, self.wv(vec))
        return ret

class FlexibleLength(nn.Module):
    def __init__(self, in_dim, out_dim, out_len, bias = False):
        super().__init__()
        self.wq = Linear(in_dim, out_len)
        self.wk = Linear(in_dim, in_dim)
        self.wv = Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.FloatTensor(out_len, out_dim))if bias else None

    def forward(self, vec):
        attn = torch.einsum('bji, bjk->bij', self.wq(vec), self.wk(vec))
        attn = torch.tanh(attn)
        attn = torch.softmax(attn, dim = -1)
        ret = torch.einsum('bij,bjk->bik', attn, self.wv(vec))
        if self.bias is not None: ret = ret + self.bias
        return ret

class FlipGradientBuilder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu    
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= -torch.log(grad_input + 1)
        # grad_input *= (1.0 / (grad_input + 1e-12))
        return grad_input

class ConvTBC1d(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, padding):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(k_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 1.0)
        self.padding = padding
    
    def forward(self, vec):
        ret = torch.conv_tbc(vec.contiguous(), self.weight, self.bias, pad = self.padding)
        return ret


















