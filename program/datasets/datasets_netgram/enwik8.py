import os, sys, time, random, warnings
from .. import *
from . import basic_netgram_dataset

@register_dataset('enwik8')
class enwik8(basic_netgram_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self._name = 'enwik8'
        self.encoding = 'utf-8'
        data_dir = 'datasets/datasets_netgram/enwik8'

        path_trn = os.path.join(data_dir, 'train.txt')
        path_val = os.path.join(data_dir, 'valid.txt')
        path_tst = os.path.join(data_dir, 'test.txt')

        self.load_dataset(path_trn, path_val, path_tst)
        self.num_label = len(self.dict)

    def build_dict_char(self, file_names, spliter, pre_eos, end_eos, lower, remove_punc):
        dict_ret = Dictionary_def()
        dict_ret.build_char_dict(upper = True)
        print(f'Dictionary constructed, len = {len(dict_ret) - dict_ret.nspecial} + {dict_ret.nspecial} = {len(dict_ret)}')
        return dict_ret

    def build_dict_word(self, file_names, spliter, pre_eos, end_eos, lower, remove_punc):
        warnings.warn('Enwik8 is a char-based dataset, character is forced to set with [True]')
        dict_ret = Dictionary_def()
        dict_ret.build_char_dict(upper = True)
        print(f'Dictionary constructed, len = {len(dict_ret) - dict_ret.nspecial} + {dict_ret.nspecial} = {len(dict_ret)}')
        return dict_ret

    def load_file_char(self, filename, spliter, pre_eos, end_eos, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        ret = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                words = self.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                tmp = [self.dict.get_index(word, unk = True) for word in words]
                tmp = np.array(tmp)
                ret.append(tmp)

        ret = np.array(ret)
        return ret
    
    def load_file_word(self, filename, spliter, pre_eos, end_eos, lower, remove_punc):
        Warning.warn('Enwik8 is a char-based dataset, character is forced to set with [True]')
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        ret = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                words = self.tokenize(line, spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                tmp = [self.dict.get_index(word, unk = False) for word in words]
                ret.append(np.array(tmp))

        ret = np.array(ret)
        return ret

    def tokenize(self, raw_line, spliter = ' ', pre_eos = False, end_eos = True, lower = True, remove_punc = False, character = True):
        raw_line = raw_line.split(' ')
        line = []
        
        for raw_char in raw_line:
            if raw_char == '' or raw_char == '\n': continue
            line.append(chr(int(raw_char)))
        line = "".join(line)

        # if lower: line = line.lower()
        # if remove_punc: line =  ''.join(c for c in line if c not in string.punctuation) # enwik8 cannot remove punctuations
        line = re.compile("\s+").sub(" ", line) # remove extra spaces ('a  b  c' -> 'a b c')
        line = line.strip()

        words = list(line) if character else line.strip().split(spliter)

        if pre_eos: words = ['<eos>'] + words
        if end_eos: words.append('<eoe>')

        return words

    @classmethod
    def setup_dataset(cls):
        return cls