import os, sys, time, random, re, json, spacy, pandas, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
# from enchant.checker import SpellChecker
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('polar')
class polar(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/rt-polaritydata'
        self.encoding = 'utf-8-sig'
        self.num_label = 2

        self.val_num = 500
        self.tst_num = 1000

        self.trn_path = os.path.join(data_dir, 'rt-polarity.neg')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'rt-polarity.pos')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def permute(self, swt_trn, swt_val, swt_tst):
        # list[np.array]
        source_trn, wizard_trn, target_trn = swt_trn
        source_val, wizard_val, target_val = swt_val# if swt_val is not None else None
        source_tst, wizard_tst, target_tst = swt_tst# if swt_tst is not None else None
        
        neg_num = len(target_trn)
        pos_num = len(target_tst)

        neg_src = source_trn
        neg_tgt = target_trn
        pos_src = source_tst
        pos_tgt = target_tst

        source = []
        target = []

        source += neg_src[:-(self.val_num + self.tst_num) // 2]
        target += neg_tgt[:-(self.val_num + self.tst_num) // 2]
        source += pos_src[:-(self.val_num + self.tst_num) // 2]
        target += pos_tgt[:-(self.val_num + self.tst_num) // 2]

        source += neg_src[-(self.val_num + self.tst_num) // 2: -self.tst_num // 2]
        target += neg_tgt[-(self.val_num + self.tst_num) // 2: -self.tst_num // 2]
        source += pos_src[-(self.val_num + self.tst_num) // 2: -self.tst_num // 2]
        target += pos_tgt[-(self.val_num + self.tst_num) // 2: -self.tst_num // 2]

        source += neg_src[-self.tst_num // 2:]
        target += neg_tgt[-self.tst_num // 2:]
        source += pos_src[-self.tst_num // 2:]
        target += pos_tgt[-self.tst_num // 2:]

        return source, None, target

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                indices = tokenizer(line)
                source.append(indices)
                target.append(0 if '.neg' in filename else 1)
        return source, None, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                words = datasets_utils.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(0 if '.neg' in filename else 1)
        return source, None, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                words = datasets_utils.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(0 if '.neg' in filename else 1)
        return source, None, target

    def build_dict_spec(self, file_names):
        dict_ret = datasets_utils.Dictionary_spec()

        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret
    
    def build_dict_index(self, file_names):
        counter = Counter()
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

            with open(file_name, 'r', encoding = self.encoding, errors='ignore') as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {file_name.split('/')[-1]}")

                    words = datasets_utils.tokenize(line, pre_eos = True, end_eos = True)
                    counter.update(words)

        dict_ret = datasets_utils.Dictionary()
        for wid, freq in counter.most_common(self.config.nvocab):
            if not wid in [dict_ret.pad, dict_ret.eos, dict_ret.eoe, dict_ret.unk]:
                dict_ret.add_word(wid, freq)
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    def build_dict_char(self, file_names):
        dict_ret = datasets_utils.build_char_dict()
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    @classmethod
    def setup_dataset(cls):
        return cls






















