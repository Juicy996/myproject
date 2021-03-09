import os, sys, time, random, torch, functools, math, json, hashlib, warnings, allennlp
import numpy as np
import torch.nn as nn
from . import datasets, models, optimizers, task, utils

class Distriibution():
    def __init__(self, hp,
                 template_config,
                 dataset_config,
        ):
        # --- config
        self.hp = hp
        self.template_config = template_config
        self.dataset_config = dataset_config

        for k, v in self.dataset_config.items():
            v.dataset = utils.load_dataset(v)
            v.config = v
            for mi in v.model_idx:
                self.criterion_config[k + f'_{mi}'].num_label = v.dataset.num_label

    def __call__(self):
        instance = None
        for k, v in self.dataset_config.items():
            instance = v.dataset
        
























