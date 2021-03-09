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

@register_model('mslstm')
class mslstm(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mslstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.net = nn.ModuleList()
        for i in range(7): 
            self.net.append(layer(self.hidden_dim))
        
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
        self.dropi = nn.Dropout(model_config.dropp)

        self.pool = None
        # self.pool = modules.FlexiblePatt(10, [1, 2, 3, 4], mode = 'avg', ceil_mode = True)
        self.pool = modules.FlexibleLQmul(10, self.hidden_dim)
        # self.fnn = nn.Linear(self.hidden_dim, 30)
        self.selector = commander4(self.net, self.hidden_dim)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        vec = modules.prepare_input(batch, embedding)

        cc = vec.clone()
        ch = vec.clone()
        dc = torch.mean(cc, dim = 1)
        dh = torch.mean(ch, dim = 1)

        selected = self.selector(vec, self.net)
        for l, cmd in zip(self.net, selected):
            cct, cht, dct, dht = l(vec, cc = cc, ch = ch, dc = dc, dh = dh)
            # cc = cc + cct * cmd
            # ch = ch + cht * cmd
            # dc = dc + dct * cmd
            # dh = dh + dht * cmd

        vec = torch.cat([ch, cc, dh[:, None, :]], dim = 1)
        
        vec = self.ff(vec)
        vec = self.dropi(vec)

        if self.pool:
            vec = self.pool(vec).reshape([source.shape[0], -1])
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
        self.tan = torch.tanh

        self.hidden_dim = hidden_dim

        self.gdtx = nn.Linear(hidden_dim, hidden_dim)
        self.gdth = nn.Linear(hidden_dim, hidden_dim)

        self.gotx = nn.Linear(hidden_dim, hidden_dim)
        self.goth = nn.Linear(hidden_dim, hidden_dim)

        self.gftx = nn.Linear(hidden_dim, hidden_dim)
        self.gfth = nn.Linear(hidden_dim, hidden_dim)

        self.f1tx = nn.Linear(hidden_dim, hidden_dim)
        self.f1th = nn.Linear(2 * hidden_dim, hidden_dim)
        self.f1ti = nn.Linear(hidden_dim, hidden_dim)
        self.f1td = nn.Linear(hidden_dim, hidden_dim)

        self.f2tx = nn.Linear(hidden_dim, hidden_dim)
        self.f2th = nn.Linear(2 * hidden_dim, hidden_dim)
        self.f2ti = nn.Linear(hidden_dim, hidden_dim)
        self.f2td = nn.Linear(hidden_dim, hidden_dim)

        self.f3tx = nn.Linear(hidden_dim, hidden_dim)
        self.f3th = nn.Linear(2 * hidden_dim, hidden_dim)
        self.f3ti = nn.Linear(hidden_dim, hidden_dim)
        self.f3td = nn.Linear(hidden_dim, hidden_dim)

        self.f4tx = nn.Linear(hidden_dim, hidden_dim)
        self.f4th = nn.Linear(2 * hidden_dim, hidden_dim)
        self.f4ti = nn.Linear(hidden_dim, hidden_dim)
        self.f4td = nn.Linear(hidden_dim, hidden_dim)

        self.fitx = nn.Linear(hidden_dim, hidden_dim)
        self.fith = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fiti = nn.Linear(hidden_dim, hidden_dim)
        self.fitd = nn.Linear(hidden_dim, hidden_dim)

        self.fotx = nn.Linear(hidden_dim, hidden_dim)
        self.foth = nn.Linear(2 * hidden_dim, hidden_dim)
        self.foti = nn.Linear(hidden_dim, hidden_dim)
        self.fotd = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vec, cc, ch, dc, dh, ctx_win = 2, mask = 1):
        shape = cc.shape
        c_ch = torch.mean(ch, dim = 1)
        e_dh = dh[:, None, :].expand(-1, shape[1], -1)

        gdt = self.sig(self.gdtx(dh) + self.gdtx(c_ch))
        got = self.sig(self.gotx(dh) + self.goth(c_ch))
        gft = self.sig(self.gftx(e_dh) + self.gfth(ch))
        # gft + mask
        # print(gft.shape)
        # input(gdt.shape)
        cat_sf = torch.nn.functional.softmax(torch.cat([gft, gdt.unsqueeze(1)], dim = 1), dim = 1)

        gft = cat_sf[:, :shape[1], :]
        gdt = torch.squeeze(cat_sf[:,  shape[1], :])

        n_dc = torch.mean(gft * cc, dim = 1) + gdt * dc
        n_dh = got * self.tan(dc)

        bef_ch = [self.get_before(ch, step + 1, self.hidden_dim) for step in range(ctx_win)]
        bef_ch = self.sum_together(bef_ch)
        aft_ch = [self.get_after(ch, step + 1, self.hidden_dim) for step in range(ctx_win)]
        aft_ch = self.sum_together(aft_ch)

        bef_cc = [self.get_before(cc, step + 1, self.hidden_dim) for step in range(ctx_win)]
        bef_cc = self.sum_together(bef_cc)
        aft_cc = [self.get_after(cc, step + 1, self.hidden_dim) for step in range(ctx_win)]
        aft_cc = self.sum_together(aft_cc)

        ctx_ch = torch.cat([bef_ch, aft_ch], axis = -1)

        e_dh = dh[:, None, :].expand([-1, shape[1], -1])
        e_dc = dc[:, None, :].expand([-1, shape[1], -1])

        f1t = self.sig(self.f1tx(ch) + self.f1th(ctx_ch) + self.f1ti(vec) + self.f1td(e_dh))
        f2t = self.sig(self.f2tx(ch) + self.f2th(ctx_ch) + self.f2ti(vec) + self.f2td(e_dh))
        f3t = self.sig(self.f3tx(ch) + self.f3th(ctx_ch) + self.f3ti(vec) + self.f3td(e_dh))
        f4t = self.sig(self.f4tx(ch) + self.f4th(ctx_ch) + self.f4ti(vec) + self.f4td(e_dh))
        i_t = self.sig(self.fitx(ch) + self.fith(ctx_ch) + self.fiti(vec) + self.fitd(e_dh))
        o_t = self.sig(self.fotx(ch) + self.foth(ctx_ch) + self.foti(vec) + self.fotd(e_dh))

        c1_5 = [torch.unsqueeze(f1t, 1), torch.unsqueeze(f2t, 1), torch.unsqueeze(f3t, 1), torch.unsqueeze(f4t, 1), torch.unsqueeze(i_t, 1)]
        # c1_5 = [f1t[:, None, :, :], f2t[:, None, :, :], f3t[:, None, :, :], f4t[:, None, :, :], i_t[:, None, :, :]]
        c1_5 = torch.nn.functional.softmax(torch.cat(c1_5, dim = 1), dim = 1)
        f1t, f2t, f3t, f4t, i_t = torch.split(c1_5, split_size_or_sections = [1, 1, 1, 1, 1], dim = 1)
        f1t, f2t, f3t, f4t, i_t = torch.squeeze(f1t), torch.squeeze(f2t), torch.squeeze(f3t), torch.squeeze(f4t), torch.squeeze(i_t)
        
        cc = f1t * bef_cc + f2t * aft_cc + f3t * vec + f4t * e_dc + i_t * cc       
        ch = o_t * self.tan(cc)
        dc = n_dc
        dh = n_dh
        
        return cc, ch, dc, dh

    def get_before(self, vec, step, hidden_dim):
        shape = vec.shape
        #padding zeros
        pre = torch.zeros([shape[0], step, hidden_dim], device = vec.device)
        #remove last steps
        rar = vec[:, :-step, :]
        #concat padding
        return torch.cat([pre, rar], dim = 1)

    def get_after(self, vec, step, hidden_dim):
        shape = vec.shape
        #padding zeros
        rar = torch.zeros([shape[0], step, hidden_dim], device = vec.device)
        #remove last steps
        pre = vec[:, step:, :]
        #concat padding
        return torch.cat([pre, rar], dim = 1)

    def sum_together(self, l):
        combined_state = None
        for tensor in l:
            if combined_state is None:
                combined_state = tensor
            else:
                combined_state = combined_state + tensor
        return combined_state

class commander4(nn.Module):
    def __init__(self, blocks, hidden_dim):
        super().__init__()
        self.conv1 = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.conv2 = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        # self.conv3 = modules.Conv1d(hidden_dim, hidden_dim, 3, padding = 1)
        self.linear1 = modules.Linear(hidden_dim, len(blocks))
        # self.linear2 = modules.Linear(128, 2)
        # self.pool = nn.AdaptiveAvgPool1d(10)
        # self.bias = nn.Parameter(torch.FloatTensor(len(blocks)))
        # nn.init.constant_(self.bias, 0.5)
    
    def forward(self, vec, blocks):
        vec = self.conv1(vec)
        # vec = torch.relu(vec)
        vec = self.conv2(vec)
        # vec = torch.relu(vec)
        # vec = self.conv3(vec)

        vec = self.linear1(vec)
        # vec = torch.relu(vec)
        # vec = self.linear2(vec)
        # vec = torch.relu(vec)
        vec = torch.mean(vec, dim = 0)
        vec = torch.mean(vec, dim = 0)
        
        # print(vec)
        # print(f'mean conv = {torch.mean(self.conv1.net.weight)}')
        # # print(self.bias)
        # vec = torch.relu(vec)
        # print(vec)
        # # print(vec.requires_grad)
        # # # input('------------------------')
        vec = torch.relu(vec)
        vec[vec > 1] = 1.0
        # input(ret.shape)
        return vec




























