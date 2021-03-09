import torch
from . import basic_model, register_model
from .. import modules, utils
import torch.nn as nn

if True:
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    except:
        from torch.nn import LayerNorm
else:
    from torch.nn import LayerNorm

@register_model('mindicator')
class mindicator(basic_model):
    def __init__(self, model_config, template_set):
        super().__init__(
            'mindicator', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.transfer = modules.Conv1d(self.embed_dim, self.hidden_dim, 3, padding = 1) if not self.embed_dim == self.hidden_dim else None
        self.dropi = nn.Dropout(model_config.dropp)

        self.net = nn.Sequential(
            Trans(),
            nn.BatchNorm1d(300),
            nn.Conv1d(300, 64, 3, padding = 0),
            
            # nn.Conv1d(64, 300, 3, padding = 0),
            # Trans(),
            # nn.Linear(256, 256),
            # Trans(),
            nn.MaxPool1d(2, 2, padding = 1),
            nn.ReLU(inplace = True),

            # nn.Conv1d(256, 128, 5, padding = 2),
            # Trans(),
            # nn.Linear(128, 128),
            # Trans(),
            # nn.MaxPool1d(2, 2, padding = 1),
            # nn.ReLU(inplace = True),

            # nn.Conv1d(128, 64, 3, padding = 1),
            # Trans(),
            # nn.Linear(64, 64),
            # Trans(),
            # nn.MaxPool1d(2, 2, padding = 1),
            # nn.ReLU(inplace = True),

            # nn.Conv1d(64, 32, 3, padding = 1),
            # Trans(),
            # nn.Linear(32, 32),
            # Trans(),
            # nn.MaxPool1d(2, 2, padding = 1),
            # Trans(),
            # nn.ReLU(inplace = True),

            Trans(),
            # nn.Linear(128, 300),
        )
        

    def forward(self, batch, embedding, extra_input, template, writer):
        vec = modules.prepare_input(batch, embedding)
        extra_output = {'to_loss': (vec, batch[-1])}
        if self.transfer: vec = self.transfer(vec)

        vec = self.net(vec)

        # writer.add_histogram(f'vec', vec, global_step = None, bins = 'tensorflow', walltime = None, max_bins = None)
        # writer.add_histogram(f'conv1', self.conv1.weight, global_step = None, bins = 'tensorflow', walltime = None, max_bins = None)
        # writer.add_histogram(f'conv2', self.conv2.weight, global_step = None, bins = 'tensorflow', walltime = None, max_bins = None)
        # writer.add_histogram(f'conv3', self.conv3.weight, global_step = None, bins = 'tensorflow', walltime = None, max_bins = None)
        # writer.add_histogram(f'conv4', self.conv4.weight, global_step = None, bins = 'tensorflow', walltime = None, max_bins = None)

        # vec = self.makeup(vec)
        vec = self.dropi(vec)
        vec = torch.mean(vec, dim = 1)

        return vec, extra_output
    
    @classmethod
    def setup_model(cls):
        return cls

class Trans(nn.Module):
    def forward(self, vec):
        return vec.transpose(1, 2)

class layer(nn.Module):
    def __init__(self, in_dim, out_dim, k_size):
        super().__init__()
        self.conv = nn.Conv1d(300, 300, k_size, padding = k_size // 2)
        # self.norm = nn.LayerNorm(300, 1e-12)
        # self.qwer = nn.Conv1d(30, 30, k_size, padding = k_size // 2)

    def forward(self, vec, residual = False):
        # vec = vec.transpose(1, 2)
        # vec = self.norm(vec)
        # vec = vec.transpose(1, 2)
        vec = self.conv(vec)
        # vec = vec.transpose(1, 2)
        vec = torch.relu(vec)

        return vec
    
    def mean(self):
        ret = torch.unsqueeze(self.fnn.net.weight, 0)
        ret = torch.cat([ret, self.conv.net.weight.permute(2, 1, 0)], dim = 0)
        # ret = torch.mean(ret, dim = 0)
        # ret = torch.mean(ret, dim = -1)
        return ret

class integrate(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = None):
        super().__init__()
        if hidden_dim is None: hidden_dim = out_dim // 4
        self.wq = nn.Linear(in_dim, out_dim)
        self.wk = nn.Linear(in_dim, out_dim)
        self.wf = nn.Linear(out_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, 1)
    def forward(self, vec):
        attn_score = torch.einsum('bki,bkj->bij', self.wq(vec), self.wk(vec))
        ret = self.wf(attn_score)
        ret = torch.relu(ret)
        ret = self.wo(ret).reshape(vec.shape[0], vec.shape[-1])
        return ret

class commander1(nn.Module):
    def __init__(self, blocks, hidden_dim):
        super().__init__()
        self.wq = modules.Linear(hidden_dim, hidden_dim)
        self.wk = modules.Linear(hidden_dim, hidden_dim)
        self.wv = modules.Linear(hidden_dim, hidden_dim)
        self.lnq = LayerNorm(hidden_dim)
        self.lnk = LayerNorm(hidden_dim)
        self.lnv = LayerNorm(hidden_dim)
    
    def forward(self, vec, block):
        weight = block.mean()
        q = self.lnq(vec)
        k = self.lnk(vec)
        v = self.lnv(weight)
        attn_score = torch.einsum('bki,bkj->bij', self.wq(q), self.wk(k))
        # attn_score = torch.sigmoid(attn_score)
        # attn_score = torch.softmax(attn_score, -1)
        ret = torch.einsum('bij,cij->bij', attn_score, self.wv(v))
        # ret = self.linear(attn_score)
        ret = torch.mean(ret)
        ret = torch.sigmoid(ret) + 0.5
        
        return ret

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
        
        print(vec)
        print(f'mean conv = {torch.mean(self.conv1.net.weight)}')
        # print(self.bias)
        vec = torch.relu(vec)
        print(vec)
        # print(vec.requires_grad)
        # # input('------------------------')
        vec = torch.relu(vec)
        # input(ret.shape)
        return vec

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
        vec = torch.nn.functional.conv1d(vec, weight, self.bias, stride = self.stride, padding = self.padding)
        return vec






