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

@register_model('normal_cnn')
class normal_cnn(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'normal_cnn', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, 3)


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

        vec = vec.transpose(1, 2)
        vec = self.conv(vec)
        vec = vec.transpose(1, 2)
        vec = torch.mean(vec, dim = 1)

        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls

# fixed, do not edit. -------------------------cagnews_91.52
@register_model('rect_cnn')
class rect_cnn(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'rect_cnn', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.net = nn.ModuleList()

        for i in range(10):
            self.net.append(block(300, 300, 3))
        
        self.fnn = modules.Linear(300, 300, activate = torch.relu)
        self.fn3 = modules.Linear(300, 30, activate = torch.relu)
        self.dropi = nn.Dropout(0.5)
        

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch

        if isinstance(embedding, nn.ModuleDict):
            vec = embedding['emb_src'](source.long())
            wiz = embedding['emb_tgt'](wizard.long())
            vec = torch.cat([wiz, vec], dim = 1)
        else:
            vec = embedding(source.long())  
        
        tmp = []
        for b in self.net:
            vec, com = b(vec)
            tmp.append(com)

        vec = self.fnn(vec)
        vec = torch.mean(vec, dim = 1, keepdim = True)
        
        cat = torch.cat(tmp, dim = -1)

        # print(vec.shape)
        # print(cat.shape)
        # input()
        cat = vec + cat
        cat = self.dropi(cat)
        cat = self.fn3(cat).reshape([-1, 300])

        # vec = self.pool(vec).reshape([-1, self.hidden_dim])
        # # vec = torch.mean(vec, dim = 1)
        vec = vec / vec.shape[-1] ** 0.5
        return cat, None
    
    @classmethod
    def setup_model(cls):
        return cls

class block(nn.Module):
    def __init__(self, in_dim, out_dim, k_size):
        super().__init__()
        self.conv = modules.Conv1d(in_dim, out_dim, k_size, residual = True)
        self.fnn  = modules.Linear(in_dim, 30)
        self.pool = modules.PostFlexiblePooling(10, [1, 2, 3, 4], mode = 'max', ceil_mode = True)

    def forward(self, vec):
        # vec = vec.transpose(1, 2)
        vec = self.conv(vec)
        # vec = vec.transpose(1, 2)
        com = self.fnn(vec)
        com = self.pool(com)
        return vec, com
#  -------------------------

@register_model('srect_cnn')
class srect_cnn(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'srect_cnn', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.task_type = 'classification'
        self.net = nn.ModuleList()

        for i in range(2):
            self.net.append(sblock(self.hidden_dim, self.hidden_dim, 3, template.shape))
        
        self.fnn = modules.Linear(self.hidden_dim, self.hidden_dim, activate = torch.relu)
        self.fn3 = modules.Linear(self.hidden_dim, 128, activate = torch.relu)
        self.dropi = nn.Dropout(0.5)
        

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
        
        tmp = []
        for b in self.net:
            vec, com = b(vec, template)
            tmp.append(com)

        vec = self.fnn(vec)
        vec = torch.mean(vec, dim = 1, keepdim = True)
        
        cat = torch.cat(tmp, dim = -1)

        # print(vec.shape)
        # print(cat.shape)
        # input()
        cat = vec + cat
        cat = self.dropi(cat)
        cat = self.fn3(cat).reshape([-1, 256])

        # vec = self.pool(vec).reshape([-1, self.hidden_dim])
        # # vec = torch.mean(vec, dim = 1)
        vec = vec / vec.shape[-1] ** 0.5
        return cat, None
    
    @classmethod
    def setup_model(cls):
        return cls

class sblock(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, template_shape):
        super().__init__()
        self.conv = modules.SConv1d(in_dim, out_dim, k_size, template_shape, padding = 1)
        self.fnn  = modules.Linear(in_dim, 128)
        self.pool = modules.PostFlexiblePooling(10, [1, 2, 3, 4], mode = 'max', ceil_mode = True)
        self.drpoi = nn.Dropout(0.5)

    def forward(self, vec, template):
        # residual = vec
        vec = vec.transpose(1, 2)
        vec = self.conv(vec, template)
        vec = vec.transpose(1, 2)
        com = self.fnn(vec)
        com = self.pool(com)

        vec = self.drpoi(vec)
        com = self.drpoi(com)
        # vec += residual
        return vec, com














