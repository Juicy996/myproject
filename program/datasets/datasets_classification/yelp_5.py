import os, sys, time, random, re, json, spacy, pandas, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
# from enchant.checker import SpellChecker
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('yelp_5')
class yelp_5(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/yelp/yelp_review_full_csv'
        self.encoding = 'utf-8-sig'
        self.num_label = 5
        self.label_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

        self.val_num = 10000
        self.tst_num = 40000

        self.trn_path = os.path.join(data_dir, 'train.csv')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'test.csv')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

        # assert self.num_label == len(set(self.target_trn.tolist() + self.target_val.tolist() + self.target_tst.tolist())), \
        #     f'not same {self.num_label} and {len(set(self.target_trn.tolist()))}'
    
    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        target = []
        csv_file = pandas.read_csv(filename, header = None)
        target_csv = csv_file[0]
        source_csv = csv_file[1]
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            indices = tokenizer(source_csv[idx])
            source.append(indices)
            target.append(self.label_dict[target_csv[idx]])

        return source, None, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename, header = None)
        target_csv = csv_file[0]
        source_csv = csv_file[1]
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            words = datasets_utils.tokenize(source_csv[idx], spliter, pre_eos, end_eos, lower, remove_punc)
            index = [self.dict.get_index(word, True) for word in words]
            source.append(torch.tensor(index))
            target.append(self.label_dict[target_csv[idx]])
        return source, wizard, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename, header = None)
        target_csv = csv_file[0]
        source_csv = csv_file[1]
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            words = datasets_utils.tokenize(source_csv[idx], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
            index = [self.dict.get_index(word, True) for word in words]
            source.append(torch.tensor(index))
            target.append(self.label_dict[target_csv[idx]])
        return source, wizard, target

    def build_dict_spec(self, file_names):
        dict_ret = datasets_utils.Dictionary_spec()

        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret
    
    def build_dict_index(self, file_names):
        counter = Counter()
        tokens = 0
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)
            csv_file = pandas.read_csv(file_name, header = None)
            tokens = 0
            target = csv_file[0]
            source = csv_file[1]
            pbar = tqdm(range(len(target)), ascii = True)
            for idx in pbar:
                pbar.set_description(f"Building dict from {file_name.split('/')[-1]}")

                words = datasets_utils.tokenize(source[idx], pre_eos = False, end_eos = False)
                counter.update(words)
                tokens += len(words)

        dictionaty = datasets_utils.Dictionary()
        for wid, freq in counter.most_common(self.config.nvocab):
            if not wid in [dictionaty.pad, dictionaty.eos, dictionaty.eoe, dictionaty.unk]:
                dictionaty.add_word(wid, freq)
        print(f'dictionaty constructed, len = {len(dictionaty) - dictionaty.special} + {dictionaty.special} = {len(dictionaty)}')
        
        return dictionaty

    def build_dict_char(self, file_names):
        dict_ret = datasets_utils.build_char_dict()
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    @classmethod
    def setup_dataset(cls):
        return cls






















