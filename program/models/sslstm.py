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

@register_model('sslstm')
class sslstm(nn.Module):
    def __init__(self, model_config, template):
        self.model_name = 'sslstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.net = nn.ModuleList()
        for i in range(model_config.nlayer): 
            self.net.append(layer(self.hidden_dim, template.shape))
        
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
        self.dropi = nn.Dropout(model_config.dropp)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch

        if isinstance(embedding, nn.ModuleDict):
            vec = embedding['emb_src'](source.long())
            wiz = embedding['emb_tgt'](wizard.long())
            vec = torch.cat([wiz, vec], dim = 1)
        else:
            vec = embedding(source.long())           
        
        

        
        shape = vec.shape
        
        
        cc = vec.clone()
        ch = vec.clone()
        dc = torch.mean(cc, dim = 1)
        dh = torch.mean(ch, dim = 1)
        

        for l in self.net:
            cc, ch, dc, dh = l(vec, cc = cc, ch = ch, dc = dc, dh = dh, template = template)
        
        vec = torch.cat([ch, cc, dh[:, None, :]], dim = 1)
        vec = self.dropi(vec)
        vec = self.ff(vec)
        vec = torch.mean(vec, dim = 1)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls

class layer(nn.Module):
    def __init__(self, hidden_dim, template_shape):
        super().__init__()
        self.sig = torch.sigmoid
        self.tan = torch.tanh

        self.hidden_dim = hidden_dim

        self.gdtx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.gdth = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.gotx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.goth = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.gftx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.gfth = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.f1tx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f1th = modules.Linear(2 * hidden_dim, hidden_dim)
        self.f1ti = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f1td = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.f2tx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f2th = modules.Linear(2 * hidden_dim, hidden_dim)
        self.f2ti = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f2td = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.f3tx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f3th = modules.Linear(2 * hidden_dim, hidden_dim)
        self.f3ti = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f3td = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.f4tx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f4th = modules.Linear(2 * hidden_dim, hidden_dim)
        self.f4ti = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.f4td = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.fitx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.fith = modules.Linear(2 * hidden_dim, hidden_dim)
        self.fiti = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.fitd = modules.SLinear(hidden_dim, hidden_dim, template_shape)

        self.fotx = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.foth = modules.Linear(2 * hidden_dim, hidden_dim)
        self.foti = modules.SLinear(hidden_dim, hidden_dim, template_shape)
        self.fotd = modules.SLinear(hidden_dim, hidden_dim, template_shape)

    def forward(self, vec, cc, ch, dc, dh, template, ctx_win = 2, mask = 1):
        shape = cc.shape
        c_ch = torch.mean(ch, dim = 1)
        e_dh = dh[:, None, :].expand(-1, shape[1], -1)

        gdt = self.sig(self.gdtx(dh, template) + self.gdtx(c_ch, template))
        got = self.sig(self.gotx(dh, template) + self.goth(c_ch, template))
        gft = self.sig(self.gftx(e_dh, template) + self.gfth(ch, template))
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

        f1t = self.sig(self.f1tx(ch, template) + self.f1th(ctx_ch) + self.f1ti(vec, template) + self.f1td(e_dh, template))
        f2t = self.sig(self.f2tx(ch, template) + self.f2th(ctx_ch) + self.f2ti(vec, template) + self.f2td(e_dh, template))
        f3t = self.sig(self.f3tx(ch, template) + self.f3th(ctx_ch) + self.f3ti(vec, template) + self.f3td(e_dh, template))
        f4t = self.sig(self.f4tx(ch, template) + self.f4th(ctx_ch) + self.f4ti(vec, template) + self.f4td(e_dh, template))
        i_t = self.sig(self.fitx(ch, template) + self.fith(ctx_ch) + self.fiti(vec, template) + self.fitd(e_dh, template))
        o_t = self.sig(self.fotx(ch, template) + self.foth(ctx_ch) + self.foti(vec, template) + self.fotd(e_dh, template))

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






























