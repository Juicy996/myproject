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

@register_model('mplstm')
class mplstm(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mplstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.net = nn.ModuleList()
        for i in range(model_config.nlayer): 
            self.net.append(layer(self.hidden_dim, self.hidden_dim))

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        vec = modules.prepare_input(batch, embedding)
        extra_output = None
        
        c_state = vec
        h_state = vec
        for idx, l in enumerate(self.net):
            ctx = vec.shape[1] // len(self.net) * (idx + 1)
            c_state, h_state = l(vec, ctx, c_state, h_state)
        
        vec = c_state + h_state
        vec = torch.mean(vec, dim = 1)
        return vec, extra_output
    
    @classmethod
    def setup_model(cls):
        return cls

class layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropp = 0):
        super().__init__()
        self.sig = torch.sigmoid
        self.tanh = torch.tanh
        self.relu = torch.relu

        self.attn = attention(in_dim, in_dim)
        self.lstm = lstm(in_dim, in_dim)
        self.fnn  = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.ReLU(inplace = True), nn.Linear(2 * in_dim, out_dim))

    def forward(self, vec, ctx, c_state, h_state):
        vec = self.fnn(vec)
        context = self.attn(vec, vec, vec, ctx)
        nc, nh = self.lstm(vec, context, c_state, h_state)
        
        return nc, nh

class attention(nn.Module):
    def __init__(self, in_dim, out_dim, dropp = 0):
        super().__init__()

        self.wq = nn.Linear(in_dim, out_dim)
        self.wk = nn.Linear(in_dim, out_dim)
        self.wv = nn.Linear(in_dim, out_dim)

        self.nq = nn.LayerNorm(in_dim)
        self.nk = nn.LayerNorm(in_dim)
        self.nv = nn.LayerNorm(in_dim)

        self.dropi = None if dropp <= 0 else nn.Dropout(dropp)

    def forward(self, q, k, v, ctx):
        mat = torch.ones([q.shape[1], q.shape[1]], device = q.device)
        mat = torch.triu(mat, -ctx) * torch.triu(mat, -ctx).t()

        attn = torch.relu(k) * v
        attn = torch.einsum('ij,bjk->bik', mat, attn)

        q = torch.relu(q) * attn
        if self.dropi: q = self.dropi(q)
        return q

class lstm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fv = nn.Linear(in_dim, out_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.fh = nn.Linear(in_dim, out_dim)

        self.dv = nn.Linear(in_dim, out_dim)
        self.dc = nn.Linear(in_dim, out_dim)
        self.dh = nn.Linear(in_dim, out_dim)

        self.iv = nn.Linear(in_dim, out_dim)
        self.ic = nn.Linear(in_dim, out_dim)
        self.ih = nn.Linear(in_dim, out_dim)

        self.ov = nn.Linear(in_dim, out_dim)
        self.oc = nn.Linear(in_dim, out_dim)
        self.oh = nn.Linear(in_dim, out_dim)

        self.sigma = torch.relu
        self.ac = torch.tanh

    def forward(self, vec, ctx, c_state, h_state):
        gf = self.sigma(self.fv(vec) + self.fc(ctx) + self.fh(h_state))
        gd = self.sigma(self.dv(vec) + self.dc(ctx) + self.dh(h_state))
        gi = self.sigma(self.iv(vec) + self.ic(ctx) + self.ih(h_state))
        go = self.sigma(self.ov(vec) + self.oc(ctx) + self.oh(h_state))

        nc = c_state * gf + gd * gi
        nh = self.ac(nc) * go
        return nc, nh

class indicator(nn.Module):
    def __init__(self, in_dim, ctx_scope, ksize, nlayer = 3):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(nlayer):
            self.net.append(nn.Conv1d(in_dim, ctx_scope if i == nlayer - 1 else in_dim, ksize))
        self.fnn1 = nn.Linear(ctx_scope, 2 * ctx_scope)
        self.fnn2 = nn.Linear(2 * ctx_scope, ctx_scope)
        self.lnv = LayerNorm(in_dim)
        self.ksize = ksize
        self.nlayer = nlayer
    
    def forward(self, q, k):
        k = self.lnv(k)
        if k.shape[1] <= q.shape[1] + self.nlayer * (self.ksize - 1):
            tmp = torch.zeros([q.shape[0], q.shape[1] + self.nlayer * (self.ksize - 1) - k, q.shape[2]], device = q.device, dtype = q.dtype)
            k = torch.cat([tmp, k], dim = 1)
        k = k.transpose(1, 2)
        for l in self.net:
            k = l(k)
            k = torch.relu(k)
        # k = torch.softmax(k, dim = 1)
        # input(k.shape)
        k = k.transpose(1, 2)
        k = self.fnn1(k)
        k = torch.relu(k)
        k = self.fnn2(k)
        k = torch.sigmoid(k)
        return k[:, -q.shape[1]:, :]

class add_transformer(nn.Module):
    def __init__(self, in_dim, out_dim, cache_len):
        super().__init__()
        self.cache_len = cache_len
        # self.lnq = nn.BatchNorm1d(in_dim, in_dim)
        # self.lnk = nn.BatchNorm1d(in_dim, in_dim)
        # self.lnv = nn.BatchNorm1d(in_dim, in_dim)

        self.lnq = LayerNorm(in_dim)
        self.lnk = LayerNorm(in_dim)
        self.lnv = LayerNorm(in_dim)

        self.wq = nn.Linear(in_dim, out_dim)
        self.wk = nn.Linear(in_dim, out_dim)
        self.wv = nn.Linear(in_dim, out_dim)

        self.hyper_net = indicator(in_dim, cache_len, 5)
    
    def forward(self, q, k, v):
        res = q

        q = self.lnq(q)
        k = self.lnk(k)
        v = self.lnv(v)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)
        # q = self.lnq(q)
        # k = self.lnk(k)
        # v = self.lnv(v)
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)

        # weight = self.hyper_net(q, k)
        weight = torch.ones([q.shape[0], q.shape[1], self.cache_len], device = q.device)
        kv = torch.relu(k) * v#torch.sigmoid(v)
        
        weight = self.integrate3(weight, kv)
        weighted = torch.einsum('bkd,bqk->bqd', kv, weight)
        q = torch.relu(q) * weighted
        # q = torch.relu(q)
        # q = q + res
        return q

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

@register_model('pgram')
class mplstm(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mplstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        self.nlayer = model_config.nlayer
        print('Model [{}] has been built.'.format(self.model_name))

        self.cache_len = 20
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
            self.rnns[idx].flatten_parameters()
            q, (nhs, ncs) = self.rnns[idx](q, None if hs is None else (hs, cs))
            q = torch.relu(q)
            k = q
            if cache is not None: 
                k = torch.cat([cache[idx], k], dim = 1)
            if k.shape[1] < q.shape[1] + self.cache_len:
                tmp = torch.zeros([q.shape[0], q.shape[1] + self.cache_len - k.shape[1], q.shape[2]], device = q.device, dtype = q.dtype)
                k = torch.cat([tmp, k], dim = 1)
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























