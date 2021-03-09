import os, sys, time, torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import register_criterion, basic_criterion
from collections import defaultdict

@register_criterion('split_loss')
class split_loss(nn.Module):
    def __init__(self, criterion_config):
        super().__init__()
        self.criterion_name = 'split_loss'
        self.hidden_dim = criterion_config.hidden_dim
        self.num_label = criterion_config.num_label
        print('Criterion [{}] has beed built.'.format(self.criterion_name))

        n_token = self.num_label
        d_embed = self.hidden_dim
        d_proj = self.hidden_dim
        div_val = criterion_config.div_val
        keep_order = criterion_config.keep_order
        cutoffs = criterion_config.cutoffs

        self.n_token = n_token
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False, extra_input = None):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''
        hidden = hidden.transpose(0, 1)
        target = target.transpose(0, 1)
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size in the batch dimension.')

        correct = 0
        total = 0

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            # indicex = target.unsqueeze(2).expand_as(logit)
            nll = -F.log_softmax(logit, dim=-1)
            # nll = nll.gather(1, indicex).squeeze(1)
            nll = torch.gather(nll.reshape([-1, self.n_token]), dim = 1, index = target.long().reshape([-1, 1]))
            correct += torch.sum(torch.argmax(logit, dim = -1) == target).cpu().numpy()
            total += target.shape[0] * target.shape[1]
            # print(correct)
            # input(total)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            correct += torch.sum(torch.argmax(head_logit, dim = -1) == target).cpu().numpy()
            total += target.shape[0] * target.shape[1]
            # print(correct)
            # input(total)

            offset = 0
            cutoff_values = [0] + self.cutoffs  # [0, 10, 50, 100, ntokens]
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)
        ret = {}
        predicts = torch.argmax(logit.detach().reshape([-1, self.num_label]), dim = 1)
        corrects = torch.eq(predicts, target.reshape([-1])).float().sum()
        total = len(target.reshape([-1]))
        ret['loss'] = nll
        ret['correct'] = corrects.unsqueeze(0).float()
        ret['total'] = torch.tensor(total, device = nll.device).unsqueeze(0).float()
        ret['loss_detach'] = nll.detach()
        ret['task_name'] = torch.tensor(1, device = nll.device).unsqueeze(0).float()
        return ret

    @classmethod
    def setup_criterion(cls):
        return cls

    