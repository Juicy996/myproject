import os, sys, time, random, re, json, spacy, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from .. import *

@register_dataset('random_classes')
class random_classes(torch.utils.data.Dataset):
    def __init__(self, dataset_config):
        super().__init__()
        self.task_type = 'test'
        self.num_label = 5
        dataset_config.num_label = self.num_label
        self.config = dataset_config
        trn_num = 25645
        val_num = 1234
        tst_num = 1347
        
        embed_dim = 300
        centers = (np.random.rand(self.num_label, embed_dim) - 0.5) * 0.5
        stds = np.random.rand(embed_dim) * 6
        self.data = {}

        source = []
        target = []
        label = 0
        for center, std in zip(centers, stds):
            for idx in range(trn_num):
                target.append(np.array([label]))
                source.append(np.random.normal(center, std, [1, embed_dim]))
            label += 1

        source_trn = torch.from_numpy(np.concatenate(source, axis = 0).astype(np.float32))
        wizard_trn = None
        target_trn = torch.from_numpy(np.concatenate(target))
        self.data['trn'] = tvt_dataset(source_trn, wizard_trn, target_trn)

        source = []
        target = []
        label = 0
        for center, std in zip(centers, stds):
            for idx in range(val_num):
                target.append(np.array([label]))
                source.append(np.random.normal(center, std, [1, embed_dim]))
            label += 1

        source_val = torch.from_numpy(np.concatenate(source, axis = 0).astype(np.float32))
        wizard_val = None
        target_val = torch.from_numpy(np.concatenate(target))
        self.data['val'] = tvt_dataset(source_val, wizard_val, target_val)

        source = []
        target = []
        label = 0
        for center, std in zip(centers, stds):
            for idx in range(tst_num):
                target.append(np.array([label]))
                source.append(np.random.normal(center, std, [1, embed_dim]))
            label += 1

        source_tst = torch.from_numpy(np.concatenate(source, axis = 0).astype(np.float32)) * 8
        wizard_tst = None
        target_tst = torch.from_numpy(np.concatenate(target))
        self.data['tst'] = tvt_dataset(source_tst, wizard_tst, target_tst)
    
    def print_self(self):
        print('hahha')

    @classmethod
    def setup_dataset(cls):
        return cls






















