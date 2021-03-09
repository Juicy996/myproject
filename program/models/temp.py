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

class indicator(nn.Module):
    def __init__(self, in_dim, ctx_scope, ksize, nlayer = 1):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(nlayer):
            self.net.append(nn.Conv1d(in_dim, ctx_scope if i == nlayer - 1 else in_dim, ksize))
        self.fnn = nn.Linear(ctx_scope, ctx_scope)
    
    def forward(self, k):
        k = k.transpose(1, 2)
        for l in self.net:
            k = l(k)
            k = torch.relu(k)
        k = k.transpose(1, 2)
        k = self.fnn(k)
        k = torch.sigmoid(k)
        return k

class add_transformer(nn.Module):
    def __init__(self, in_dim, out_dim, cache_len):
        super().__init__()
        self.cache_len = cache_len

        self.lnq = LayerNorm(in_dim)
        self.lnk = LayerNorm(in_dim)
        self.lnv = LayerNorm(in_dim)

        self.wq = nn.Linear(in_dim, out_dim)
        self.wk = nn.Linear(in_dim, out_dim)
        self.wv = nn.Linear(in_dim, out_dim)

        self.hyper_kernel = 5
        self.hyper_layer = 1
        self.hyper_net = indicator(in_dim, cache_len, self.hyper_kernel, self.hyper_layer)
        
    def forward(self, query, key, value):
        ret = []
        input(query.shape)
        k = self.lnk(key)
        v = self.lnv(value)

        k = self.wk(k)
        k = torch.relu(k)
        k = k * self.wv(v)

        for idx in range(query.shape[1] - (self.hyper_kernel - 1) * self.hyper_layer):
            q = query[:, idx : idx + self.hyper_kernel, :]

            q = self.lnq(q)
            q = self.wq(q)
            torch.relu(q)

            weight = self.hyper_net(q).squeeze().unsqueeze(-1)
            
            tmp = k[:, idx: idx + self.cache_len, :]#[:, None, :]
            tmp = tmp * weight
            ret.append(torch.mean(tmp, dim =1, keepdim = True))

        
        ret = torch.cat(ret, dim = 1)
        input(ret.shape)
        return ret

    def integrate3(self, mat, vec):
        mshape = mat.shape
        vshape = vec.shape
        sup = vshape[1] - mshape[1] - mshape[2] + 1
        assert sup >= 0

        mat = torch.cat([mat, torch.zeros([*mshape[:2], mshape[1]], device = vec.device, dtype = vec.dtype)], dim = 2)
        if sup > 0: mat = torch.cat([torch.zeros([*mshape[:2], sup], device = vec.device, dtype = vec.dtype), mat], dim = 2)
        
        mat = mat.reshape([mshape[0], -1])[:, :mshape[1] * vshape[1]].reshape([*mshape[:2], vshape[1]])
        return mat

class Boom(nn.Module):
    def __init__(self, in_dim, hidden_dim = -1, out_dim = -1, dropp = 0.1, shortcut = False):
        super(Boom, self).__init__()
        if hidden_dim == -1: hidden_dim = 2 * in_dim
        if out_dim == -1: out_dim = in_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = None if shortcut else nn.Linear(hidden_dim, out_dim)
        self.dropi = nn.Dropout(dropp) if dropp else None
        self.act = GELU()

    def forward(self, vec):
        ret = self.linear1(vec)
        ret = self.act(ret)
        if self.dropi: ret = self.dropi(ret)
        if self.linear2:
            ret = self.linear2(ret)
        else:
            in_dim = vec.shape[2]
            ret = torch.narrow(x, 2, 0, ret.shape[2] // in_dim * in_dim)
            ret = ret.view(*ret.shape[:-1], ret.shape[2] // in_dim, in_dim)
            ret = ret.sum(dim = 2)            

        return ret

class GELU(nn.Module):
    """
    BERT used the GELU instead of RELU
    """
    def forward(self, x):
        #return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)

@register_model('temp')
class mplstm(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mplstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        self.nlayer = model_config.nlayer
        print('Model [{}] has been built.'.format(self.model_name))

        self.cache_len = 200
        self.dropi = nn.Dropout(model_config.dropp)

        self.atts = nn.ModuleList()
        self.fnns = nn.ModuleList()
        self.rnns = nn.ModuleList()
        self.rnln = nn.ModuleList()
        for i in range(model_config.nlayer): 
            self.atts.append(add_transformer(self.hidden_dim, self.hidden_dim, self.cache_len))
            self.fnns.append(Boom(self.hidden_dim))
            self.rnns.append(nn.LSTM(self.hidden_dim, self.hidden_dim, model_config.nlayer, batch_first = True))
            self.rnln.append(LayerNorm(self.hidden_dim))

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        extra_output = {'cache': [], 'hstas': [], 'cstas': []}
        vec = modules.prepare_input(batch, embedding)
        cache = None
        if extra_input is not None and 'cache' in extra_input:
            cache = extra_input['cache']
        hstas = None
        if extra_input is not None and 'hstas' in extra_input:
            hstas = extra_input['hstas']
        cstas = None
        if extra_input is not None and 'cstas' in extra_input:
            cstas = extra_input['cstas']
        
        q = vec
        for idx in range(self.nlayer):
            hs = None if hstas is None else hstas[idx]
            cs = None if cstas is None else cstas[idx]
            q = self.rnln[idx](q)
            q, (nhs, ncs) = self.rnns[idx](q, None if hs is None else (hs, cs))
            q = torch.relu(q)
            k = q
            if cache is not None: 
                k = torch.cat([cache[idx], k], dim = 1)
            if k.shape[1] < q.shape[1] + self.cache_len:
                tmp = torch.zeros([q.shape[0], q.shape[1] + self.cache_len - k.shape[1], q.shape[2]], device = q.device, dtype = q.dtype)
                k = torch.cat([tmp, k], dim = 1)
            assert k.shape[1] == q.shape[1] + self.cache_len
            extra_output['cache'].append(k[:, -self.cache_len:, :])
            extra_output['hstas'].append(nhs)
            extra_output['cstas'].append(ncs)
            v = k

            q = self.atts[idx](q, k, v)
            q = self.fnns[idx](q)

            if self.dropi: q = self.dropi(q)
        
        return q, extra_output
    
    @classmethod
    def setup_model(cls):
        return cls
























