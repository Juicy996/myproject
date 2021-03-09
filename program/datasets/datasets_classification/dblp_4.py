import os, sys, time, random, re, torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .. import register_dataset, datasets_utils
from . import basic_label_dataset

class dblp_4(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                line_tmp = line.replace('\t', ' ').split(' ')
                indices = tokenizer(' '.join(line_tmp[1:]))
                source.append(indices)
                target.append(self.label_set[line_tmp[0]])
        return source, None, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                line_tmp = line.replace('\t', ' ').split(' ')
                words = datasets_utils.tokenize(' '.join(line_tmp[1:]), spliter, pre_eos, end_eos, lower, remove_punc)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(self.label_set[line_tmp[0]])
        return source, None, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                line_tmp = line.replace('\t', ' ').split(' ')
                words = datasets_utils.tokenize(' '.join(line_tmp[1:]), spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(self.label_set[line_tmp[0]])
        return source, None, target

    def build_dict_spec(self, file_names):
        dict_ret = datasets_utils.Dictionary_spec()
        # tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        # for file_name in file_names:
        #     if file_name is None: continue
        #     assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

        #     with open(file_name, 'r', encoding = self.encoding) as f:
        #         lines = tqdm(f.readlines(), ascii = True)
        #         for line in lines:
        #             lines.set_description(f"Building dict from {file_name.split('/')[-1]}")
        #             indices = tokenizer(line[1:])
        #             words = 'eos ' + line + 'eoe'
        #             dict_ret.add_line(words, indices)

        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret
    
    def build_dict_index(self, file_names):
        counter = Counter()
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

            with open(file_name, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {file_name.split('/')[-1]}")

                    words = datasets_utils.tokenize(line[1:], pre_eos = True, end_eos = True)
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

@register_dataset('dblp_database')
class dblp_database(dblp_4):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/dblp_4'
        self.encoding = 'utf-8-sig'
        self.label_set = {'ICDE': 0, 'SIGMODConference': 1, 'VLDB': 2, 'SIGMODRecord': 3}
        self.num_label = len(self.label_set)
        self.balance = True

        self.val_num = 500
        self.tst_num = 1000

        self.trn_path = os.path.join(data_dir, 'database.txt')
        self.val_path = None
        self.tst_path = None

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        # np.array[torch.tensor]

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('dblp_datamining')
class dblp_datamining(dblp_4):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/dblp_4'
        self.label_set = {'KDD': 0, 'PKDD': 1, 'ICDM': 2, 'SDM': 3}
        self.num_label = len(self.label_set)

        self.val_num = 220
        self.tst_num = 440

        self.trn_path = os.path.join(data_dir, 'datamining.txt')
        self.val_path = None
        self.tst_path = None

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('dblp_theory')
class dblp_theory(dblp_4):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/dblp_4'
        self.label_set = {'FOCS': 0, 'SODA': 1, 'STOC': 2, 'FOCS': 3}
        self.num_label = len(self.label_set)

        self.val_num = 400
        self.tst_num = 800

        self.trn_path = os.path.join(data_dir, 'theory.txt')
        self.val_path = None
        self.tst_path = None

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('dblp_visualization')
class dblp_visualization(dblp_4):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/dblp_4'
        self.label_set = {'ICCV': 0, 'IEEEVisualization': 1, 'CVPR': 2, 'VAST': 3, 'IEEETVCG': 4, 'InformationVisualization': 5}
        self.num_label = len(self.label_set)

        self.val_num = 400
        self.tst_num = 800

        self.trn_path = os.path.join(data_dir, 'visualization.txt')
        self.val_path = None
        self.tst_path = None

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

















