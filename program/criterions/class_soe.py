import os, sys, time, torch
# import torch.nn as nn
from . import register_criterion, basic_criterion
# from torch.nn.modules.loss import _Loss



@register_criterion('class_soe')
class class_soe(basic_criterion):
    def __init__(self, criterion_config):
        super().__init__('class_soe', criterion_config.hidden_dim, criterion_config.num_label)
        self.lbd = criterion_config.lbd
    
    # def __init__(self, rep_dim, num_label):
    #     super().__init__('classification', rep_dim, num_label)

    def forward(self, rep, target, reduce = True, extra_input = None):
        source, target = extra_input['to_loss']
        logits = self.fnn(rep)
        logits = logits.reshape([-1, self.num_label])
        target = target.reshape(-1).long()

        predicts = torch.argmax(logits.detach(), dim = 1)
        corrects = torch.eq(predicts, target).float().sum()
        total = len(target)

        # loss = torch.nn.functional.nll_loss(logits, target, size_average = False, ignore_index = self.padding_idx, reduce = reduce)
        loss = torch.nn.functional.cross_entropy(logits, target)#, reduction = 'sum', ignore_index = self.padding_idx

        loss += self.pen(rep, predicts, source, target)

        ret = {}
        ret['loss'] = loss.unsqueeze(0)
        ret['correct'] = corrects.detach().unsqueeze(0)
        ret['total'] = torch.tensor(total, device = loss.device, dtype = self.fnn[0].weight.dtype).detach().unsqueeze(0)
        ret['loss_detach'] = loss.detach().unsqueeze(0)

        return ret

    def pen(self, rep, predict, src, targets):
        src = torch.mean(src, dim = 1)
        
        cen_rep, cen_src = [], []
        for i in range(self.num_label):
            cen_rep.append(torch.mean(rep[torch.eq(predict, i)], dim = 0) if i in predict else torch.zeros([self.hidden_dim], device = rep.device))
            cen_src.append(torch.mean(src[torch.eq(predict, i)], dim = 0) if i in targets else torch.zeros([self.hidden_dim], device = rep.device))

        cen_rep = torch.cat(cen_rep).reshape([-1, self.num_label])
        cen_src = torch.cat(cen_src).reshape([-1, self.num_label])

        k_ = self.knn(cen_src, 1)
        s_ = self.soe(cen_rep)

        ret = k_ * s_
        ret = torch.mean(ret)
        return ret

    def knn(self, centers, k):
        v1 = centers.unsqueeze(0).repeat(centers.shape[0], 1, 1)
        v2 = centers.unsqueeze(1).repeat(1, centers.shape[0], 1)

        dis = (v1 - v2) ** 2
        dis = torch.sum(dis, dim = -1)
        topk = torch.topk(dis, 2, dim = -1, largest = False, sorted = True, out = None)[0][:, 0].unsqueeze(1)
        zero = torch.tensor(0, dtype = torch.float, device = centers.device)
        one  = torch.tensor(1, dtype = torch.float, device = centers.device)
        ret = torch.where(dis > topk, zero, one)

        ret1 = ret.unsqueeze(0).repeat(ret.shape[0], 1, 1)
        ret2 = ret.unsqueeze(1).repeat(1, ret.shape[0], 1)
        ret = ret1 * ret2
        return ret
 
    def soe(self, centers):
        # b1 = a.unsqueeze(0).repeat(a.shape[0], 1, 1)
        # b2 = a.unsqueeze(1).repeat(1, a.shape[0], 1)
        # b3 = a.unsqueeze(1).repeat(1, a.shape[0], 1)

        # ab = (b1 - b2) **2
        # ac = (b1 - b3) **2

        # ab = ab.unsqueeze(2).repeat(1, 1, a.shape[0], 1)
        # ac = ac.unsqueeze(1).repeat(1, a.shape[0], 1, 1)
        # br = ab - ac + lbd
        b1 = centers.unsqueeze(0).unsqueeze(0).repeat(centers.shape[0], centers.shape[0], 1, 1)
        b2 = centers.unsqueeze(0).unsqueeze(2).repeat(centers.shape[0], 1, centers.shape[0], 1)
        b3 = centers.unsqueeze(1).unsqueeze(2).repeat(1, centers.shape[0], centers.shape[0], 1)

        br = (b1 - b2) ** 2 - (b1 - b3) ** 2 + self.lbd
        br = br.permute(2,1,0,3)

        br = torch.sum(torch.relu(br) ** 2, dim = -1)
        return br

    @classmethod
    def setup_criterion(cls):
        return cls

    