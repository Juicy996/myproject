import os, sys, torch, re
import numpy as np
import warnings
from tqdm import tqdm
from collections import Counter
from .. import *

class basic_netgram_dataset(basic_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.shuffle = True
        self.task_type = 'n'
        self.nvocab = dataset_config.nvocab
        self.seq_len = dataset_config.seq_len

    def build_dict_char(self, file_names, spliter, pre_eos, end_eos, lower, remove_punc):
        counter = Counter()
        for filename in file_names:
            assert os.path.exists(filename), 'File [{}] doesn\'t exists.'.format(filename)
            with open(filename, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Processing {filename.split('/')[-1]}")
                    chars = self.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                    counter.update(chars)

        dict_ret = Dictionary_def()
        for wid, freq in counter.most_common(self.nvocab):
            if not wid in [dict_ret.pad, dict_ret.eos, dict_ret.eoe, dict_ret.unk]:
                dict_ret.add_word(wid, freq)
        print(f'Dictionary constructed, len = {len(dict_ret) - dict_ret.nspecial} + {dict_ret.nspecial} = {len(dict_ret)}')
        return dict_ret

    def build_dict_word(self, file_names, spliter, pre_eos, end_eos, lower, remove_punc):
        counter = Counter()
        for filename in file_names:
            assert os.path.exists(filename), 'File [{}] doesn\'t exists.'.format(filename)
            with open(filename, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Processing {filename.split('/')[-1]}")
                    words = self.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = False)
                    counter.update(words)

        dict_ret = Dictionary_def()
        for wid, freq in counter.most_common(self.nvocab):
            if not wid in [dict_ret.pad, dict_ret.eos, dict_ret.eoe, dict_ret.unk]:
                dict_ret.add_word(wid, freq)
        print(f'Dictionary constructed, len = {len(dict_ret) - dict_ret.nspecial} + {dict_ret.nspecial} = {len(dict_ret)}')
        return dict_ret

    def load_file_char(self, filename, spliter, pre_eos, end_eos, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        ret = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                chars = self.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                tmp = [self.dict.get_index(word, unk = False) for word in chars]
                ret.append(np.array(tmp))

        ret = np.array(ret)
        return ret
    
    def load_file_word(self, filename, spliter, pre_eos, end_eos, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        ret = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                words = self.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = False)
                tmp = [self.dict.get_index(word, unk = False) for word in words]
                ret.append(np.array(tmp))

        ret = np.array(ret)
        return ret

    def load_dataset(self, file_trn, file_val, file_tst):
        self.dict = self.build_dict([file_trn, file_val, file_tst])

        trn = np.concatenate(self.load_file(file_trn))
        self.trn_num = len(trn)
        self.trn = seq_dataset(trn)
        
        val = np.concatenate(self.load_file(file_val))
        self.val_num = len(val)
        self.val = seq_dataset(val)
        
        tst = np.concatenate(self.load_file(file_tst))
        self.tst_num = len(tst)
        self.tst = seq_dataset(tst)
        

        if 'glove' in self.pretrained:
            self.embedding_matrix, self.missing = self.get_embedding_matrix(self.dict) 
        else:
            self.embedding_matrix, self.missing = None, 0
        print('Dataset [{}] has been built.'.format(self._name))
    
    def build_dict(self, 
                   file_names, 
                   spliter = ' ', 
                   pre_eos = False, 
                   end_eos = False,
                   lower = True,
                   remove_punc = True,
        ):
        if self.embedding_type in ['word', 'glove_word']:
            ret = self.build_dict_word
        elif self.embedding_type in ['char', 'glove_char']:
            ret = self.build_dict_char
        else:
            raise ValueError(f'[{self._name}] doesn\'t supported pretrained mode [{self.embedding_type}]')

        return ret(file_names, spliter, pre_eos, end_eos, lower, remove_punc)

    def load_file(self, 
                  filename, 
                  spliter = ' ', 
                  pre_eos = False, 
                  end_eos = False,
                  lower = True,
                  remove_punc = True,
        ):
        if self.embedding_type in ['word', 'glove_word']:
            ret = self.load_file_word
        elif self.embedding_type in ['char', 'glove_char']:
            ret = self.load_file_char
        else:
            raise ValueError(f'[{self._name}] doesn\'t supported pretrained mode [{self.embedding_type}]')

        return ret(filename, spliter, pre_eos, end_eos, lower, remove_punc)

    def print_self(self):
        print(f'The length of dictionary is{len(self.dict)}')
        print(f'Train token = {self.trn_num}, valid token = {self.val_num}, test token = {self.tst_num}')




























