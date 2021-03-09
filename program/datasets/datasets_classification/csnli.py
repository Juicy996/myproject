import os, sys, time, random, re, json
import numpy as np
from tqdm import tqdm
from collections import Counter
from .. import *
from . import basic_label_dataset

@register_dataset('csnlim')
class csnlim(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self._name = 'csnlim'
        data_dir = 'datasets/datasets_multi/snli_1.0'
        self.encoding = 'utf-8'
        self.dict_label = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        self.num_label = len(self.dict_label)
        self.trn_path = os.path.join(data_dir, 'snli_1.0_train.jsonl')
        self.val_path = os.path.join(data_dir, 'snli_1.0_dev.jsonl')
        self.tst_path = os.path.join(data_dir, 'snli_1.0_test.jsonl')

        self.val_num = 10000
        self.tst_num = 10000
        
        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def load_file_transformer(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.pretrained)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                dt = json.loads(line)

                tmp_line = dt['sentence1'] + ' | ' + dt['sentence2']
                tmp_labe = dt['gold_label']
                if not tmp_labe in self.dict_label:
                    if 'train' in filename: continue
                    elif 'dev' in filename: self.val_num = self.val_num - 1
                    elif 'test' in filename: self.tst_num = self.tst_num - 1
                    else: raise ValueError('Unknown error')
                    continue

                indices = tokenizer(tmp_line)
                source.append(indices.numpy())
                target.append(self.dict_label[tmp_labe])
        return source, wizard, target

    def load_file_word(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                dt = json.loads(line)

                tmp_labe = dt['gold_label']
                if not tmp_labe in self.dict_label:
                    if 'train' in filename: continue
                    elif 'dev' in filename: self.val_num = self.val_num - 1
                    elif 'test' in filename: self.tst_num = self.tst_num - 1
                    else: raise ValueError('Unknown error')
                    continue

                wordss = self.tokenize(dt['sentence1'], spliter, pre_eos, end_eos, lower, remove_punc)
                wordsw = self.tokenize(dt['sentence2'], spliter, pre_eos, end_eos, lower, remove_punc)
                indexs = [self.dict.get_index(word, True) for word in wordss]
                indexw = [self.dict.get_index(word, True) for word in wordsw]
                source.append(np.array(indexs + indexw))
                target.append(self.dict_label[tmp_labe])
        return source, wizard, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                dt = json.loads(line)

                tmp_labe = dt['gold_label']
                if not tmp_labe in self.dict_label:
                    if 'train' in filename: continue
                    elif 'dev' in filename: self.val_num = self.val_num - 1
                    elif 'test' in filename: self.tst_num = self.tst_num - 1
                    else: raise ValueError('Unknown error')
                    continue

                wordss = self.tokenize(dt['sentence1'], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                wordsw = self.tokenize(dt['sentence2'], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                indexs = [self.dict.get_index(word, True) for word in wordss]
                indexw = [self.dict.get_index(word, True) for word in wordsw]
                source.append(np.array(indexs + indexw))
                target.append(self.dict_label[tmp_labe])
        return source, wizard, target
    
    def build_dict_word(self, file_names):
        counter = Counter()
        for filename in file_names:
            if filename is None: continue
            assert os.path.exists(filename), 'File [{}] doesn\'t exists.'.format(file_name)

            with open(filename, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {filename.split('/')[-1]}")
                    dt = json.loads(line)

                    tmp_labe = dt['gold_label']
                    if not tmp_labe in self.dict_label:
                        if 'train' in filename: continue
                        elif 'dev' in filename: self.val_num = self.val_num - 1
                        elif 'test' in filename: self.tst_num = self.tst_num - 1
                        else: raise ValueError('Unknown error')
                        continue
                
                    counter.update(self.tokenize(dt['sentence1'], pre_eos = False, end_eos = False))
                    counter.update(self.tokenize(dt['sentence2'], pre_eos = False, end_eos = False))

        dictionaty = Dictionary_def()
        for wid, freq in counter.most_common(self.nvocab):
            if not wid in [dictionaty.pad, dictionaty.eos, dictionaty.eoe, dictionaty.unk]:
                dictionaty.add_word(wid, freq)
        print(f'dictionaty constructed, len = {len(dictionaty) - dictionaty.nspecial} + {dictionaty.nspecial} = {len(dictionaty)}')
        
        return dictionaty

    def build_dict_char(self, file_names):
        dict_ret = Dictionary_def()
        dict_ret.build_char_dict()
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    def build_dict_transformer(self, file_names):
        dict_ret = Dictionary_emp()
        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('csnlic')
class csnlic(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self._name = 'csnlic'
        data_dir = 'datasets/datasets_multi/snli_1.0'
        self.encoding = 'utf-8'
        self.dict_label = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        self.num_label = len(self.dict_label)
        self.trn_path = os.path.join(data_dir, 'snli_1.0_train.jsonl')
        self.val_path = os.path.join(data_dir, 'snli_1.0_dev.jsonl')
        self.tst_path = os.path.join(data_dir, 'snli_1.0_test.jsonl')

        self.val_num = 10000
        self.tst_num = 10000
        
        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def load_file_transformer(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.pretrained)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                dt = json.loads(line)

                tmp_line1 = dt['sentence1']
                tmp_line1 = dt['sentence2']
                tmp_labe = dt['gold_label']
                if not tmp_labe in self.dict_label:
                    if 'train' in filename: continue
                    elif 'dev' in filename: self.val_num = self.val_num - 1
                    elif 'test' in filename: self.tst_num = self.tst_num - 1
                    else: raise ValueError('Unknown error')
                    continue

                indices = tokenizer(tmp_line)
                source.append(indices.numpy())
                target.append(self.dict_label[tmp_labe])
        return source, wizard, target

    def load_file_word(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                dt = json.loads(line)

                tmp_labe = dt['gold_label']
                if not tmp_labe in self.dict_label:
                    if 'train' in filename: pass
                    elif 'dev' in filename: self.val_num = self.val_num - 1
                    elif 'test' in filename: self.tst_num = self.tst_num - 1
                    else: raise ValueError('Unknown error')
                    continue

                wordss = self.tokenize(dt['sentence1'], spliter, pre_eos, end_eos, lower, remove_punc)
                wordsw = self.tokenize(dt['sentence2'], spliter, pre_eos, end_eos, lower, remove_punc)
                indexs = [self.dict.get_index(word, True) for word in wordss]
                indexw = [self.dict.get_index(word, True) for word in wordsw]
                source.append(np.array(indexs + indexw))
                target.append(self.dict_label[tmp_labe])
        return source, wizard, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                dt = json.loads(line)

                tmp_labe = dt['gold_label']
                if not tmp_labe in self.dict_label:
                    if 'train' in filename: pass
                    elif 'dev' in filename: self.val_num = self.val_num - 1
                    elif 'test' in filename: self.tst_num = self.tst_num - 1
                    else: raise ValueError('Unknown error')
                    continue

                wordss = self.tokenize(dt['sentence1'], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                wordsw = self.tokenize(dt['sentence2'], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                indexs = [self.dict.get_index(word, True) for word in wordss]
                indexw = [self.dict.get_index(word, True) for word in wordsw]
                source.append(np.array(indexs + indexw))
                target.append(self.dict_label[tmp_labe])
        return source, wizard, target
    
    def build_dict_word(self, file_names):
        counter = Counter()
        for filename in file_names:
            if filename is None: continue
            assert os.path.exists(filename), 'File [{}] doesn\'t exists.'.format(filename)

            with open(filename, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {filename.split('/')[-1]}")
                    dt = json.loads(line)

                    tmp_labe = dt['gold_label']
                    if not tmp_labe in self.dict_label:
                        if 'train' in filename: pass
                        elif 'dev' in filename: self.val_num = self.val_num - 1
                        elif 'test' in filename: self.tst_num = self.tst_num - 1
                        else: raise ValueError('Unknown error')
                        continue
                
                    counter.update(self.tokenize(dt['sentence1'], pre_eos = False, end_eos = False))
                    counter.update(self.tokenize(dt['sentence2'], pre_eos = False, end_eos = False))

        dictionaty = Dictionary_def()
        for wid, freq in counter.most_common(self.nvocab):
            if not wid in [dictionaty.pad, dictionaty.eos, dictionaty.eoe, dictionaty.unk]:
                dictionaty.add_word(wid, freq)
        print(f'dictionaty constructed, len = {len(dictionaty) - dictionaty.nspecial} + {dictionaty.nspecial} = {len(dictionaty)}')
        
        return dictionaty

    def build_dict_char(self, file_names):
        dict_ret = Dictionary_def()
        dict_ret.build_char_dict()
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    def build_dict_transformer(self, file_names):
        dict_ret = Dictionary_emp()
        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret

    @classmethod
    def setup_dataset(cls):
        return cls





















