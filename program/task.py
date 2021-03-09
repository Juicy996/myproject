import os, sys, time, random, importlib, torch, hashlib, warnings
from . import criterions, modules, models
import torch.nn as nn

class Task(torch.nn.Module):
    def __init__(self, 
                 dataset,
                 template_config, 
                 model_config, 
                 criterion_config,
        ):
        super().__init__()
        # dataset
        self.dataset = dataset
        # parameter template
        self.template = None if template_config['is_none'] else self.create_template(template_config)
        # --- Build Embeddings
        self.embedding_set = self.create_embeddings(dataset)
        # Build models
        self.model_set = self.create_models(model_config)
        # --- Build Criterions
        self.criterion_set = self.create_criterions(criterion_config)

    def create_template(self, template_config_set):
        tmp_config = template_config_set['t1']
        tmp_ret = torch.FloatTensor(1, tmp_config.template_count, tmp_config.nrow, tmp_config.ncol)
        getattr(torch.nn.init, v.template_initializer)(tmp_ret)
        temp_ret = nn.Parameter(tmp_ret)

        return temp_ret

    def create_models(self, model_config):
        model_set = nn.ModuleDict()
        for k, v in model_config.items():
            model_set[k] = models.load_model(v.model_name)(v, self.template)
        return model_set

    def create_embeddings(self, dataset):
        # dataset: dataset_config
        embedding_set = nn.ModuleDict()
        for name, config in dataset.items():
            if config.dataset.task_type in ['c', 'n', 'd']:
                embedding_set[name] = modules.Embedding(ntoken = len(config.dataset.dict), 
                                                        embed_dim = config.embed_dim, 
                                                        pretrained_matrix = config.dataset.embedding_matrix, 
                                                        trainable = config.embedding_trainable,
                                                       )

            elif config.dataset.task_type in ['t', 'm']:
                emb_src = modules.Embedding(ntoken = len(config.dataset.dict_src), 
                                            embed_dim = config.embed_dim, 
                                            pretrained_matrix = config.dataset.embedding_matrix_src, 
                                            trainable = config.embedding_trainable,
                                           )
                emb_tgt = modules.Embedding(ntoken = len(config.dataset.dict_wiz), 
                                            embed_dim = config.embed_dim, 
                                            pretrained_matrix = config.dataset.embedding_matrix_wiz, 
                                            trainable = config.embedding_trainable,
                                           )
                embedding_set[name] = nn.ModuleDict({'emb_src': emb_src, 'emb_tgt': emb_tgt})

            else:
                warnings.warn(f'Unknown dataset_type [{config.dataset.task_type}], embedding will be set with None.')
                embedding_set[name] = None

        return embedding_set

    def create_criterions(self, criterion_config):
        criterion_set = nn.ModuleDict()
        for k, v in criterion_config.items():
            criterion_set[k] = criterions.load_criterion(v.criterion_name)(v)
        return criterion_set
    
    def forward(self, data_config, batch, extra_input, model_idx, writer):
        rep, extra_output = self.model_set[model_idx](batch, self.embedding_set[data_config.dataset._name], extra_input, self.template, writer)
        if isinstance(rep, dict):
            ret_dict = rep
        else:
            ret_dict = self.criterion_set[f'{data_config.dataset._name}_{model_idx}'](rep, batch[-1], extra_input = extra_output)
        ret_dict['extra_output'] = extra_output
        return ret_dict





























