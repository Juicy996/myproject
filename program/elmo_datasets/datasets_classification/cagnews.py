import os, sys, time, random, re, json, spacy, torch
import numpy as np
from tqdm import tqdm
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('cagnews')
class cagnews(basic_label_dataset):
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

    
    def load_file(self,
                  filename, 
                  spliter = ' ', 
                  pre_eos = True, 
                  end_eos = True,
                  lower = True,
                  remove_punc = True,
        ):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                dt = json.loads(line)

                wordss = self.tokenize(dt['text'], spliter, pre_eos, end_eos, lower, remove_punc)
                wordsw = self.tokenize(dt['headline'], spliter, pre_eos, end_eos, lower, remove_punc)
                source.append(torch.cat([wordsw, wordss], dim = 1))

                target.append(int(dt['label'] - 1))
        return source, wizard, target

    @classmethod
    def setup_dataset(cls):
        return cls






















