import os, sys, time, random, re, json, spacy, pandas, torch, warnings
import numpy as np
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
# from enchant.checker import SpellChecker
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('sst2_np')
class sst2_np(basic_label_dataset):
    def __init__(self, dataset_config):
        warnings.warn(f"dataset_config will be forced to None since [SST2] is an index - index dataset.")
        dataset_config.pretrained = None
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/SST2'
        self.encoding = 'utf-8-sig'
        self.num_label = 2

        self.val_num = 600
        self.tst_num = 1221

        self.trn_path = os.path.join(data_dir, 'train_array.npy')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'test_array.npy')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                indices = tokenizer(line[1:])
                source.append(indices)
                target.append(int(line[0]))
        return source, None, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source_tmp = list(np.load(filename)[:, 1:])
        target = list(np.load(filename)[:, 0])
        source = []

        lines = tqdm(source_tmp, ascii = True)
        for line in lines:
            lines.set_description(f"Processing {filename.split('/')[-1]}")
            index = [self.dict.get_index(word, True) for word in line]
            source.append(torch.tensor(index))     

        return source, None, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                words = datasets_utils.tokenize(line[1:], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(int(line[0]))
        return source, None, target

    def build_dict_spec(self, file_names):
        raise ValueError('This dataset [SST2] doesn\'t support transformer pretrained mode.')
    
    def build_dict_index(self, file_names):
        counter = Counter()
        dict_ret = datasets_utils.Dictionary()
        dict_ret.pop('<pad>')
        dict_ret.pop('<unk>')
        dict_ret.pop('<eos>')
        dict_ret.pop('<eoe>')

        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

            source = np.load(file_name)[:, 1:]
            lines = tqdm(source, ascii = True)
            for line in lines:
                lines.set_description(f"Building dict from {file_name.split('/')[-1]}")
                counter.update(line)

        for wid, freq in counter.most_common(self.config.nvocab):
            dict_ret.add_word(wid, freq)
        print(f'Dictionary was set to transform index to index and nvocab was became useless since this '
                f'dataset was loaded from numpy files, len = {len(dict_ret)}.')
        return dict_ret

    def build_dict_char(self, file_names):
        raise ValueError('This dataset [SST2] dosen\'t support character mode.')

    @classmethod
    def setup_dataset(cls):
        return cls



















