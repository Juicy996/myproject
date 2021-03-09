import os, sys, time, random, re, json, spacy
import numpy as np
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
# from enchant.checker import SpellChecker
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('magnews')
class agnews(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_multi/AGnews'
        self.encoding = 'utf-8'
        self.num_label = 4

        self.trn_path = os.path.join(data_dir, 'train.jsonl')
        self.val_path = os.path.join(data_dir, 'dev.jsonl')
        self.tst_path = os.path.join(data_dir, 'test.jsonl')

        self.val_num = 5000
        self.tst_num = 7600
        
        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

        self.target_trn -= 1
        self.target_val -= 1
        self.target_tst -= 1

        assert self.num_label == len(set(self.target_trn.tolist()))
    
    def load_file(self,
                  filename, 
                  spliter = ' ', 
                  pre_eos = True, 
                  end_eos = True,
                  lower = True,
                  remove_punc = True,
                  # must return list
        ):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = []
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                dt = json.loads(line)

                words = datasets_utils.tokenize(dt['text'], spliter, pre_eos, end_eos, lower, remove_punc)
                index = [self.dict[0].get_index(word, True) for word in words]
                source.append(np.array(index))

                words = datasets_utils.tokenize(dt['headline'], spliter, pre_eos, end_eos, lower, remove_punc)
                index = [self.dict[1].get_index(word, True) for word in words]
                wizard.append(np.array(index))

                target.append(int(dt['label']))
        return source, wizard, target

    def build_dict(self, file_names: [str]) -> datasets_utils.Dictionary:
        counter_src = Counter()
        counter_tgt = Counter()
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)
            with open(file_name, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {file_name.split('/')[-1]}")
                    if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                    dt = json.loads(line)
                    counter_src.update(datasets_utils.tokenize(dt['text'], pre_eos = False, end_eos = False))
                    counter_tgt.update(datasets_utils.tokenize(dt['headline'], pre_eos = False, end_eos = False))

        dict_src = datasets_utils.Dictionary()
        for wid, freq in counter_src.most_common(self.config.nvocab_src):
            if not wid in [dict_src.pad, dict_src.eos, dict_src.eoe, dict_src.unk]:
                dict_src.add_word(wid, freq)
        print(f'dict_src constructed, len = {len(dict_src) - dict_src.special} + {dict_src.special} = {len(dict_src)}')
        
        dict_tgt = datasets_utils.Dictionary()
        for wid, freq in counter_tgt.most_common(self.config.nvocab_tgt):
            if not wid in [dict_tgt.pad, dict_tgt.eos, dict_tgt.eoe, dict_tgt.unk]:
                dict_tgt.add_word(wid, freq)
        print(f'dict_tgt constructed, len = {len(dict_tgt) - dict_tgt.special} + {dict_tgt.special} = {len(dict_tgt)}')
        return [dict_src, dict_tgt]

    @classmethod
    def setup_dataset(cls):
        return cls






















