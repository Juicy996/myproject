import os, sys, torch
import numpy as np
from .. import basic_dataset
from ... import modules

class basic_label_dataset(basic_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.task_type = 'c'
    
    def load_dataset(self, file_trn: str, file_val: str = None, file_tst: str = None):
        self.source_trn, self.wizard_trn, self.target_trn, \
        self.source_val, self.wizard_val, self.target_val, \
        self.source_tst, self.wizard_tst, self.target_tst = self.basic_load_dataset(file_trn, file_val, file_tst)

    def permute(self, swt_trn, swt_val, swt_tst):
        # list[np.array]
        source_trn, wizard_trn, target_trn = swt_trn
        source_val, wizard_val, target_val = swt_val# if swt_val is not None else None
        source_tst, wizard_tst, target_tst = swt_tst# if swt_tst is not None else None

        source = source_trn
        if source_val is not None: source += source_val
        if source_tst is not None: source += source_tst

        wizard = wizard_trn
        if wizard_val is not None: wizard += wizard_val
        if wizard_tst is not None: wizard += wizard_tst

        target = target_trn
        if target_val is not None: target += target_val
        if target_tst is not None: target += target_tst

        return source, wizard, target

    

    
    




















