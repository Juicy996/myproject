import torch
from . import basic_model, register_model
from .. import modules
import torch.nn as nn
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

@register_model('mlstm')
class mlstm(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mlstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.net = nn.ModuleList()
        for i in range(model_config.nlayer):
            self.net.append(nn.LSTM(self.hidden_dim, self.hidden_dim, model_config.nlayer, batch_first = True))
        self.dropi = nn.Dropout(0.5)


    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = modules.prepare_input(batch, embedding)
        if self.transfer: vec = self.transfer(vec)
        extra_output = {'hstate': [], 'cstate': []}

        for idx, lstm in enumerate(self.net):
            if extra_input is not None and 'hstate' in extra_input:
                hs = extra_input['hstate'][idx]
                cs = extra_input['cstate'][idx]
            else:
                hs, cs = None, None
            vec, (nhs, ncs) = lstm(vec, None if hs is None else (hs, cs))
            vec = torch.relu(vec)
            extra_output['hstate'].append(nhs)
            extra_output['cstate'].append(ncs)

        # vec = self.dropi(vec)
        return vec, extra_output
    
    @classmethod
    def setup_model(cls):
        return cls

@register_model('mcnn')
class ncnn(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mcnn'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))
        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.net = nn.ModuleList()
        kernels = [5, 5, 5]
        for i in range(model_config.nlayer):
            k_size = 3
            self.net.append(
                nn.Sequential(
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding = k_size // 2),
                    # nn.MaxPool1d(2, 2),
                    nn.ReLU()
                )
            )

        self.fnn = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        vec = modules.prepare_input(batch, embedding)
        if self.transfer: vec = self.transfer(vec)
        extra_output = {'to_loss': [vec, target]}

        vec = vec.transpose(1, 2)

        for idx, l in enumerate(self.net):
            vec = l(vec)

        vec = vec.transpose(1, 2)

        vec = self.fnn(vec)
        return vec, extra_output

    
    @classmethod
    def setup_model(cls):
        return cls

@register_model('mbilstm')
class mbilstm(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mbilstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.net = nn.LSTM(self.hidden_dim, self.hidden_dim, model_config.nlayer, batch_first = True, bidirectional = True)
        self.dropi = nn.Dropout(0.5)


    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = modules.prepare_input(batch, embedding)
        if self.transfer: vec = self.transfer(vec)
        extra_output = {'to_loss': [vec, target]}

        vec, (h_n, c_n) = self.net(vec)

        vec = h_n[-1, :, :]

        vec = self.dropi(vec)
        return vec, extra_output
    
    def context(self, vec, window, alpha = 0.2):
        window = min(vec.shape[1] - 1, window)
        bef = torch.zeros(vec.shape, device = vec.device, dtype = vec.dtype)
        aft = torch.zeros(vec.shape, device = vec.device, dtype = vec.dtype)
        # bef = vec.clone().detach()
        # aft = vec.clone().detach()
        for idx in range(window):
            vec = vec * alpha
            pad = torch.zeros(vec.shape[0], idx  + 1, vec.shape[2], device = vec.device, dtype = vec.dtype)
            lft_pad = torch.cat([pad, vec], dim = 1)[:, :vec.shape[1], :]
            bef += lft_pad
            rit_pad = torch.cat([vec, pad], dim = 1)[:, -vec.shape[1]:, :]
            aft += rit_pad
        return bef, aft
    
    def Bef_aft_n(self, len, alpha = 0.2, window = 1):
        bef = np.zeros([len,len],dtype = np.float32)
        bef = np.tril(bef, 0)

        for i in range(len):
            factor = 1.0
            bound = min(i + window + 1, len)
            for j in range(i, bound):
                bef[j, i] = factor
                factor *= alpha
        for i in range(len):
            bef[i,i] = 0.0
        aft = bef.copy().T
        return torch.tensor(bef), torch.tensor(aft)

    
    @classmethod
    def setup_model(cls):
        return cls

@register_model('mlinear')
class linear(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mbilstm'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))
        self.net = nn.Linear(self.embed_dim, self.hidden_dim)
    
    def forward(self, batch, embedding, extra_input, template, writer):
        # input(batch[0][0])
        vec = modules.prepare_input(batch, embedding)
        # vec = self.net(vec)
        vec = torch.relu(vec)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls

@register_model('direct')
class direct(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, batch, embedding, extra_input, template, writer):
        vec = modules.prepare_input(batch, embedding)
        return vec, None
    
    @classmethod
    def setup_model(cls):
        return cls

@register_model('maskcnn')
class ncnn(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mcnn'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))
        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        
        self.net = nn.Conv1d(self.hidden_dim, self.hidden_dim, 5)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        vec = modules.prepare_input(batch, embedding)
        if self.transfer: vec = self.transfer(vec)
        extra_output = None

        vec = torch.cat([torch.zeros([vec.shape[0], 4, vec.shape[2]], device = vec.device, dtype = vec.dtype), vec], dim = 1)
        vec = vec.transpose(1, 2)

        vec = self.net(vec)
        vec = torch.relu(vec)

        vec = vec.transpose(1, 2)

        return vec, extra_output

    
    @classmethod
    def setup_model(cls):
        return cls










































