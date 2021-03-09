import os, sys
import numpy as np
from .. import *

class basic_label_dataset(basic_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.shuffle = True
        self.task_type = 'c'
        self.nvocab = dataset_config.nvocab
    
    def load_dataset(self, file_trn: str, file_val: str = None, file_tst: str = None):
        self.dict = self.build_dict([file_trn, file_val, file_tst])

        swt_trn = self.load_file(file_trn) 
        swt_val = self.load_file(file_val) if file_val is not None else [None, None, None]
        swt_tst = self.load_file(file_tst) if file_tst is not None else [None, None, None]

        source, wizard, target = self.permute(swt_trn, swt_val, swt_tst)
        if hasattr(self, 'balance') and self.balance: source, wizard, target = self.process_balance(source, wizard, target)

        source_trn  = source[:-(self.val_num + self.tst_num)]
        wizard_trn  = wizard[:-(self.val_num + self.tst_num)] if wizard is not None else None
        target_trn  = target[:-(self.val_num + self.tst_num)]

        source_val  = source[-(self.val_num + self.tst_num) : -self.tst_num]
        wizard_val  = wizard[-(self.val_num + self.tst_num) : -self.tst_num] if wizard is not None else None
        target_val  = target[-(self.val_num + self.tst_num) : -self.tst_num]

        source_tst  = source[-self.tst_num:]
        wizard_tst  = wizard[-self.tst_num:] if wizard is not None else None
        target_tst  = target[-self.tst_num:]

        self.src_trn_len, self.src_val_len, self.src_tst_len = self.avg_len([source_trn, source_val, source_tst])
        self.wiz_trn_len, self.wiz_val_len, self.wiz_tst_len = self.avg_len([wizard_trn, wizard_val, wizard_tst]) if wizard_trn is not None else [0, 0, 0]
        self.tgt_trn_dic, self.tgt_val_dic, self.tgt_tst_dic = self.target_statisic([target_trn, target_val, target_tst])
        self.target_set_trn, self.target_set_val, self.target_set_tst = set(target_trn), set(target_val), set(target_tst)
        self.trn_num, self.val_num, self.tst_num = len(source_trn), len(source_val), len(source_tst)

        self.trn = st_dataset(source_trn, wizard_trn, target_trn)
        self.val = st_dataset(source_val, wizard_val, target_val)
        self.tst = st_dataset(source_tst, wizard_tst, target_tst)

        if 'glove' in self.pretrained:
            self.embedding_matrix, self.missing = self.get_embedding_matrix(self.dict) 
        else:
            self.embedding_matrix, self.missing = None, 0

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
    
    def process_balance(self, source, wizard, target):
        print(f'Balance mode will probably disturb the order of original dataset.')
        lable_kind = set(target)
        src_dict = {}
        wiz_dict = None if wizard is None or len(wizard) == 0 else {}
        cnt_dict = {}
        total = len(target)
        for i in lable_kind:
            src_dict[i] = []
            if wiz_dict is not None: wiz_dict[i] = []
            cnt_dict[i] = 0
        for idx, tgt in enumerate(target):
            src_dict[tgt].append(source[idx])
            if wiz_dict is not None: wiz_dict[tgt].append(wizard[idx])
            cnt_dict[tgt] += 1

        trn_nums = {}
        val_nums = {}
        tst_nums = {}
        for k, v in cnt_dict.items():
            val_nums[k] = math.floor(cnt_dict[k] // 10)
            tst_nums[k] = int(val_nums[k] * 2)
            trn_nums[k] = cnt_dict[k] - val_nums[k] - tst_nums[k]

        source_ret = []
        wizard_ret = None if wizard is None or len(wizard) == 0 else []
        target_ret = []
        for k, v in sorted(cnt_dict.items(), key = lambda kv:(kv[1], kv[0])):
            source_ret += src_dict[k][:trn_nums[k]]
            if wizard is not None: wizard_ret += wiz_dict[k][:trn_nums[k]]
            target_ret += [k] * trn_nums[k]
        
        for k, v in sorted(cnt_dict.items(), key = lambda kv:(kv[1], kv[0])):
            source_ret += src_dict[k][trn_nums[k]: -tst_nums[k]]
            if wizard is not None: wizard_ret += wiz_dict[k][trn_nums[k]: -tst_nums[k]]
            target_ret += [k] * val_nums[k]

        for k, v in sorted(cnt_dict.items(), key = lambda kv:(kv[1], kv[0])):
            source_ret += src_dict[k][-tst_nums[k]:]
            if wizard is not None: wizard_ret += wiz_dict[k][-tst_nums[k]:]
            target_ret += [k] * tst_nums[k]

        assert len(source_ret) == len(source), f'source len error, got source_ret = {len(source_ret)} and source = {len(source)}'
        assert len(target_ret) == len(target), f'source len error, got target_ret = {len(target_ret)} and target = {len(target)}'
        return source_ret, wizard_ret, target_ret
    
    def print_self(self):
        print(f'The length of dictionary is{len(self.dict)}')
        print(f'Train num = {self.trn_num}, valid num = {self.val_num}, test num = {self.tst_num}')

        print(f'Average length of training source = {self.src_trn_len}')
        print(f'Average length of validation source = {self.src_val_len}')
        print(f'Average length of test source = {self.src_tst_len}')

        print(f'Average length of training wizard = {self.wiz_trn_len}')
        print(f'Average length of validation wizard = {self.wiz_trn_len}')
        print(f'Average length of test wizard = {self.wiz_trn_len}')

        if self.task_type in ['c']:
            print(f'Training targets = {self.tgt_trn_dic}')
            print(f'Validation targets = {self.tgt_val_dic}')
            print(f'Test targets = {self.tgt_tst_dic}')
            print(f'Class Index_trn = {self.target_set_trn}')
            print(f'Class Index_val = {self.target_set_val}')
            print(f'Class Index_tst = {self.target_set_tst}')

    def build_dict(self, file_names):
        if self.embedding_type in ['word', 'glove_word']:
            ret = self.build_dict_word
        elif self.embedding_type in ['char', 'glove_char']:
            ret = self.build_dict_char
        elif self.embedding_type in ['transformer_word']:
            ret = self.build_dict_transformer
        else:
            raise ValueError(f'Unsupported pretrained mode [{self.embedding_type}]')

        return ret(file_names)

    def load_file(self, 
                  filename, 
                  spliter = ' ', 
                  pre_eos = True, 
                  end_eos = True,
                  verbose = True,
                  lower = True,
                  remove_punc = True,
        ):
        if self.embedding_type in ['word', 'glove_word']:
            ret = self.load_file_word
        elif self.embedding_type in ['char', 'glove_char']:
            ret = self.load_file_char
        elif self.embedding_type in ['transformer_word']:
            ret = self.load_file_transformer
        else:
            raise ValueError(f'Unsupported pretrained mode [{self.embedding_type}]')

        return ret(filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc)



















