import os, sys, pandas, torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset


@register_dataset('spam')
class spam(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/spam'
        self.num_label = 2
        # self.encoding = 'utf-8'
        self.val_num = 500
        self.tst_num = 1000
        self.trn_path = os.path.join(data_dir, 'spam.csv')
        self.val_path = None
        self.tst_path = None
        self.dict_label = {'ham': 0, 'spam': 1}
        # self.dict = self.build_dict(self.trn_path)
        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename, index_col=False)
        target_csv = csv_file['v1']
        source_csv = csv_file['v2']
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            indices = tokenizer(source_csv[idx])
            source.append(indices)
            target.append(self.dict_label[target_csv[idx]])
        return source, wizard, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename, index_col=False)
        target_csv = csv_file['v1']
        source_csv = csv_file['v2']
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            words = datasets_utils.tokenize(source_csv[idx], spliter, pre_eos, end_eos, lower, remove_punc)
            index = [self.dict.get_index(word, True) for word in words]
            source.append(torch.tensor(index))
            target.append(int(self.dict_label[target_csv[idx]]))
        return source, wizard, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename, index_col=False)
        target_csv = csv_file['v1']
        source_csv = csv_file['v2']
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            words = datasets_utils.tokenize(source_csv[idx], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
            index = [self.dict.get_index(word, True) for word in words]
            source.append(torch.tensor(index))
            target.append(int(self.dict_label[target_csv[idx]]))
        return source, wizard, target

    def build_dict_spec(self, file_names):
        dict_ret = datasets_utils.Dictionary_spec()

        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret
    
    def build_dict_index(self, file_names):
        counter = Counter()
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

            csv_data = pandas.read_csv(file_name, header=0)
            target_csv = csv_data['v1']
            source_csv = csv_data['v2']
            pbar = tqdm(range(len(target_csv)), ascii = True)
            for idx in pbar:
                pbar.set_description(f"Building dict from {file_name.split('/')[-1]}")
                words = datasets_utils.tokenize(source_csv[idx], spliter=' ', pre_eos=False, end_eos=True, lower=True, remove_punc=False)
                counter.update(words)

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
