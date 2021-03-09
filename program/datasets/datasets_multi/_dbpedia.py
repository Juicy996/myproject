import os, sys, time, random, re, json, spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('dbpedia')
class dbpedia(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_multi/dbpedia_csv'
        self.encoding = 'utf-8'
        self.num_label = 14

        self.val_num = 60000
        self.tst_num = 70000

        self.trn_path = os.path.join(data_dir, 'train.csv')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'test.csv')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

        self.target_trn -= 1
        self.target_val -= 1
        self.target_tst -= 1

        assert self.num_label == len(set(self.target.tolist()))
    
    def load_file(self,
                  filename, 
                  dictionary, 
                  unk,
                  encoding,
                  spliter = ' ', 
                  pre_eos = True, 
                  end_eos = True,
                  verbose = True,
                  lower = True,
                  remove_punc = True,
        ):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = []
        target = []
        idx = 0
        csv_file = pd.read_csv(filename, header = None)
        labels = csv_file[0]
        titles = csv_file[1]
        texts  = csv_file[2]

        pbar = tqdm(range(len(labels)), ascii = True)
        for idx in pbar:
            pbar.set_description(f"Processing {filename.split('/')[-1]}")
            words = datasets_utils.tokenize(titles[idx], spliter, pre_eos, end_eos, lower, remove_punc)
            index = [dictionary[1].get_index(word, True) for word in words]
            wizard.append(np.array(index))

            words = datasets_utils.tokenize(texts[idx], spliter, pre_eos, end_eos, lower, remove_punc)
            index = [dictionary[0].get_index(word, True) for word in words]
            source.append(np.array(index))

            target.append(int(labels[idx]))

        return source, wizard, target

    def build_dict(self, file_names, encoding):
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)
            
            csv_data = pd.read_csv(file_name, header = None)
            titles = csv_data[1]
            texts  = csv_data[2]

            counter_src = Counter()
            counter_tgt = Counter()
            tokens_src = 0
            tokens_tgt = 0
            pbar = tqdm(range(len(titles)), ascii = True)
            for idx in pbar:
                pbar.set_description(f"Building dict from {file_name.split('/')[-1]}")

                words = datasets_utils.tokenize(texts[idx], pre_eos = False, end_eos = False)
                tokens_src += len(words)
                counter_src.update(words)

                words = datasets_utils.tokenize(titles[idx], pre_eos = False, end_eos = False)
                tokens_tgt += len(words)
                counter_tgt.update(words)

        dict_src = datasets_utils.Dictionary()
        for wid, freq in counter_src.most_common(self.nvocab_src):
            if not wid in [dict_src.pad, dict_src.eos, dict_src.eoe, dict_src.unk]:
                dict_src.add_word(wid, freq)
        print(f'dict_src constructed, len = {len(dict_src) - dict_src.special} + {dict_src.special} = {len(dict_src)}')
        
        dict_tgt = datasets_utils.Dictionary()
        for wid, freq in counter_tgt.most_common(self.nvocab_tgt):
            if not wid in [dict_tgt.pad, dict_tgt.eos, dict_tgt.eoe, dict_tgt.unk]:
                dict_tgt.add_word(wid, freq)
        print(f'dict_tgt constructed, len = {len(dict_tgt) - dict_tgt.special} + {dict_tgt.special} = {len(dict_tgt)}')
        return dict_src, dict_tgt

    def load_dataset(self, file_trn, file_val = None, file_tst = None, shuffle = False):
        list_tmp = [file_trn]
        if file_val is not None: list_tmp.append(file_val)
        if file_tst is not None: list_tmp.append(file_tst)

        self.dict = self.build_dict(list_tmp, self.encoding)

        source_trn, wizard_trn, target_trn = self.load_file(file_trn, self.dict, False, self.encoding)
        source_tst, wizard_tst, target_tst = self.load_file(file_tst, self.dict, False, self.encoding)
        
        source_trn = np.array(source_trn, dtype = object)
        wizard_trn = np.array(wizard_trn, dtype = object)
        target_trn = np.array(target_trn                )

        source_tst = np.array(source_tst, dtype = object)
        wizard_tst = np.array(wizard_tst, dtype = object)
        target_tst = np.array(target_tst                )

        self.source_trn = source_trn
        self.wizard_trn = wizard_trn
        self.target_trn = target_trn

        self.source_val = np.concatenate([source_tst[5000 * i: 5000 * i + 1000] for i in range(14)])
        self.wizard_val = np.concatenate([wizard_tst[5000 * i: 5000 * i + 1000] for i in range(14)])
        self.target_val = np.concatenate([target_tst[5000 * i: 5000 * i + 1000] for i in range(14)])

        self.source_tst = np.concatenate([source_tst[5000 * i + 1000: 5000 * (i +1)] for i in range(14)])
        self.wizard_tst = np.concatenate([wizard_tst[5000 * i + 1000: 5000 * (i +1)] for i in range(14)])
        self.target_tst = np.concatenate([target_tst[5000 * i + 1000: 5000 * (i +1)] for i in range(14)])

        if self.pretrained is not None:
            embedding_matrix_src, missing_src = datasets_utils.get_embedding_matrix(self.embed_dim, self.dict[0], self.pretrained)
            embedding_matrix_tgt, missing_tgt = datasets_utils.get_embedding_matrix(self.embed_dim, self.dict[1], self.pretrained)
        else:
           embedding_matrix_src, missing_src = None, 0
           embedding_matrix_tgt, missing_tgt = None, 0
        self.embedding_matrix = (embedding_matrix_src, embedding_matrix_tgt)
        print(f'Train num = {len(self.target_trn)}, valid num = {len(self.target_val)}, test num = {len(self.target_tst)}')
        print(f'Dataset [{self.dataset_name}] has been built, missing = [{missing_src} and {missing_tgt}].')

    @classmethod
    def setup_dataset(cls):
        return cls






















