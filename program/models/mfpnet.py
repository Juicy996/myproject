import torch
from . import basic_model, register_model
from .. import modules, criterions#, models
import torch.nn as nn
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

@register_model('mfpnet')
class mfpnet(nn.Module):
    def __init__(self, model_config, template):
        super().__init__()
        self.model_name = 'mfpnet'
        self.embed_dim = model_config.embed_dim
        self.hidden_dim = model_config.hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

        self.net1 = mfpnet1(model_config, 4)
        self.net2 = mfpnet2(model_config, 4)

        self.sigma1 = torch.nn.Parameter(torch.FloatTensor(torch.tensor([10.5])))
        self.sigma2 = torch.nn.Parameter(torch.FloatTensor(torch.tensor([0.5])))

    def forward(self, batch, embedding, extra_input, template, writer):
        ret1, rep = self.net1(batch, embedding, None, None, None)
        ret2 = self.net2(batch, embedding, rep, None, None)

        loss1 = ret1['loss']
        loss2 = ret2['loss']

        loss = loss1 / (self.sigma1 ** 2 + 1e-12) + loss2 / (self.sigma2 ** 2 + 1e-12) + torch.log(self.sigma1) + torch.log(self.sigma2)
        ret2['loss'] = loss
        ret2['loss_detach'] = loss.detach()
        # print(f'sigma1 = {self.sigma1}')
        # print(f'loss1 = {loss1}')
        # print(f'sigma2 = {self.sigma2}')
        # print(f'loss2 = {loss2}')
        return ret2, None  

    @classmethod
    def setup_model(cls): return cls

class mfpnet2(nn.Module):
    def __init__(self, hp, num_label):
        super().__init__()
        self.embed_dim = hp.embed_dim
        self.hidden_dim = hp.hidden_dim
        self.num_label = num_label

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        # self.pm = modules.PositionalEmbedding(self.hidden_dim)
        self.net = b_class(hp.hidden_dim, hp.hidden_dim)
        self.fnn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.criterion = cross_entropy(self.hidden_dim, self.num_label)
        self.dropi = nn.Dropout(0.5)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = embedding(source.long())
        if self.transfer: vec = self.transfer(vec)
        # extra_output = {'to_loss': [vec, target]}

        # vec = self.pm(vec) # positional embedding
        vec, _ = self.net([vec, None, None], None, None, None, None)
        vec = self.fnn(vec)
        vec = torch.relu(vec)

        d = self.nb_alg(vec, extra_input)
        vec = vec - self.nb_alg(vec, d).reshape(vec.shape)

        vec = self.linear(vec)
        vec = torch.relu(vec)
        vec = self.dropi(vec)
        vec = torch.mean(vec, dim = 1)
        ret_dict = self.criterion(vec, target)
        return ret_dict
    
    def nb_alg(self, u, v):
        shape = u.shape
        u = u.reshape([shape[0], -1])
        v = v.reshape([shape[0], -1])
        mag = (torch.sum(v ** 2, axis = -1) ** 0.5).reshape([shape[0],1])
        normalized = (1.0 / mag)  * v
        projection = u * u * normalized

        return projection

class mfpnet1(nn.Module):
    def __init__(self, hp, num_label):
        super().__init__()
        self.embed_dim = hp.embed_dim
        self.hidden_dim = hp.hidden_dim
        self.num_label = num_label

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        # self.pm = modules.PositionalEmbedding(self.hidden_dim)
        self.net = b_class(hp.hidden_dim, hp.hidden_dim)
        self.fnn = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropi = nn.Dropout(0.5)
        self.criterion = max2min_loss(self.hidden_dim, self.num_label)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = embedding(source.long())
        if self.transfer: vec = self.transfer(vec)
        # extra_output = {'to_loss': [vec, target]}

        # vec = self.pm(vec) # positional embedding
        vec, _ = self.net([vec, None, None], None, None, None, None)
        vec = self.fnn(vec)
        rep = torch.relu(vec)

        vec = modules.FlipGradientBuilder.apply(rep)
        vec = self.linear(rep)
        vec = torch.relu(vec)
        vec = self.dropi(vec)
        vec = torch.mean(vec, dim = 1)
        ret_dict = self.criterion(vec, target)
        return ret_dict, rep

class max2min_loss(nn.Module):    
    def __init__(self, rep_dim, num_label):
        super().__init__()
        self.fnn = nn.Linear(rep_dim, num_label)
        self.hidden_dim = rep_dim
        self.num_label = num_label

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        # loss = torch.nn.functional.nll_loss(logits, target, size_average = False, ignore_index = self.padding_idx, reduce = reduce)
        loss = torch.nn.functional.cross_entropy(logits, target)#, reduction = 'sum', ignore_index = self.padding_idx

        ret = {}
        ret['loss'] = loss.unsqueeze(0)
        ret['correct'] = corrects.detach().unsqueeze(0)
        ret['total'] = torch.tensor(total, device = loss.device, dtype = loss.dtype).detach().unsqueeze(0)
        ret['loss_detach'] = loss.detach().unsqueeze(0)
        ret['rep'] = logits.detach()
        return ret

class cross_entropy(nn.Module):    
    def __init__(self, rep_dim, num_label):
        super().__init__()
        self.fnn = nn.Linear(rep_dim, num_label)
        self.hidden_dim = rep_dim
        self.num_label = num_label

    def forward(self, rep, target, reduce = True, extra_input = None):
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        # loss = torch.nn.functional.nll_loss(logits, target, size_average = False, ignore_index = self.padding_idx, reduce = reduce)
        loss = torch.nn.functional.cross_entropy(logits, target)#, reduction = 'sum', ignore_index = self.padding_idx
        loss = 1.0 / loss

        ret = {}
        ret['loss'] = loss.unsqueeze(0)
        ret['correct'] = corrects.detach().unsqueeze(0)
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn.weight.dtype).detach().unsqueeze(0)
        ret['loss_detach'] = loss.detach().unsqueeze(0)
        ret['rep'] = logits.detach()
        return ret

class b_class(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.net = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding = 1),
            # nn.MaxPool1d(2, 2),
            nn.ReLU(inplace = True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding = 1),
            # nn.MaxPool1d(2, 2),
            nn.ReLU(inplace = True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding = 1),
            nn.ReLU(inplace = True),
        )
        self.dropi = nn.Dropout(0.5)


    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = modules.prepare_input(batch, embedding)
        if self.transfer: vec = self.transfer(vec)
        extra_output = {'to_loss': [vec, target]}

        vec = vec.transpose(1, 2)
        vec = self.net(vec)
        vec = vec.transpose(1, 2)

        vec = self.dropi(vec)
        # ret = torch.mean(vec, dim = 1)

        return vec, extra_output


















