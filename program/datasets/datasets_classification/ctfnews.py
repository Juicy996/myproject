import os, sys, pandas, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset


@register_dataset('ctfnews')
class ctfnews(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_multi/news'
        self.num_label = 2
        self.encoding = 'utf-8'
        self.val_num = 4500
        self.tst_num = 9000
        self.trn_path = os.path.join(data_dir, 'True.csv')
        self.val_path = os.path.join(data_dir, 'Fake.csv')
        self.tst_path = None

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def permute(self, swt_trn, swt_val, swt_tst):
        # list[np.array]
        source_t, wizard_t, target_t = swt_trn
        source_f, wizard_f, target_f = swt_val

        source_trn = source_t[:15699] + source_f[:15699]
        target_trn = target_t[:15699] + target_f[:15699]

        source_val = source_t[15699: 15699 + self.val_num // 2] + source_f[15699: 15699 + self.val_num // 2]
        target_val = target_t[15699: 15699 + self.val_num // 2] + target_f[15699: 15699 + self.val_num // 2]

        source_tst = source_t[-self.tst_num // 2:] + source_f[-self.tst_num // 2:]
        target_tst = target_t[-self.tst_num // 2:] + target_f[-self.tst_num // 2:]

        source = source_trn + source_val + source_tst
        target = target_trn + target_val + target_tst

        return source, None, target

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename)
        title_csv = csv_file['title']
        text_csv = csv_file['text']
        subject_csv = csv_file['subject']
        tf = [1] if 'True' in filename else [0]
        target = tf * len(title_csv)
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            line = title_csv[idx] + '|' + text_csv[idx] + '|' + subject_csv[idx]
            indices = tokenizer(line)
            source.append(indices)
        return source, wizard, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename)
        title_csv = csv_file['title']
        text_csv = csv_file['text']
        subject_csv = csv_file['subject']
        tf = [1] if 'True' in filename else [0]
        target = tf * len(title_csv)
        pbar = tqdm(range(len(text_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            line = title_csv[idx] + '|' + text_csv[idx] + '|' + subject_csv[idx]
            words = datasets_utils.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc)
            index = [self.dict.get_index(word, True) for word in words]
            source.append(torch.tensor(index))
        return source, wizard, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        csv_file = pandas.read_csv(filename)
        title_csv = csv_file['title']
        text_csv = csv_file['text']
        subject_csv = csv_file['subject']
        tf = [1] if 'True' in filename else [0]
        target = tf * len(title_csv)
        pbar = tqdm(range(len(target_csv)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            line = title_csv[idx] + '|' + text_csv[idx] + '|' + subject_csv[idx]
            words = datasets_utils.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = True)
            index = [self.dict.get_index(word, True) for word in words]
            source.append(torch.tensor(index))
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
            text_csv = csv_data['text']
            title_csv = csv_data['title']
            pbar = tqdm(range(len(text_csv)), ascii = True)
            for idx in pbar:
                pbar.set_description(f"Building dict from {file_name.split('/')[-1]}")
                line = title_csv[idx] + '|' + text_csv[idx]
                words = datasets_utils.tokenize(line, spliter=' ', pre_eos=False, end_eos=True, lower=True, remove_punc=False)
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
