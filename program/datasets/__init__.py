import os, sys, time, random, importlib, torch, math, re, string
from collections import Counter
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

dataset_dict = {}
dataset_names = ['datasets_translation', 'datasets_classification', 'datasets_netgram', 'datasets_multi']

class st_dataset(torch.utils.data.Dataset):
    def __init__(self, source, wizard, target):
        super().__init__()
        self.source = source
        self.target = target

    def __getitem__(self, index):
        src = self.source[index]
        tgt = self.target[index]
        return [src, torch.tensor(0), tgt]

    def __len__(self):
        return len(self.target)

class seq_dataset(torch.utils.data.Dataset):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.fixed = False

    def __getitem__(self, index):
        if not hasattr(self, 'batch_size'): raise RuntimeError('Equip function should be executed before this operation.')
        src = self.source[index]
        tgt = self.target[index]
        return [src, torch.tensor(0), tgt]

    def __len__(self):
        return len(self.source)

    def equip(self, batch_size, seq_len):
        if self.fixed: raise RuntimeError('Batch_size and seq_len of this dataset can not be edited once equipped, you must reload the dataset.')
        self.fixed = True

        self.batch_size = batch_size
        self.seq_len = seq_len

        nbatch = len(self.source) // (self.batch_size * self.seq_len)
        self.source = self.source[: nbatch * self.batch_size * self.seq_len + 1]
        self.target = self.source[1:]
        self.source = self.source[: -1]

        self.source = self.source.reshape([self.batch_size, -1, self.seq_len]).transpose(1, 0, 2).reshape([-1, self.seq_len])
        self.target = self.target.reshape([self.batch_size, -1, self.seq_len]).transpose(1, 0, 2).reshape([-1, self.seq_len])
        return self

class swt_dataset(torch.utils.data.Dataset):
    def __init__(self, source, wizard, target):
        super().__init__()
        self.source = source
        self.wizard = wizard
        self.target = target

    def __getitem__(self, index):
        src = self.source[index]
        wiz = self.wizard[index]
        tgt = self.target[index]
        return [src, wiz, tgt]

    def __len__(self):
        return len(self.target)

class basic_dataset():
    def __init__(self, dataset_config):
        self.transformers_dict = ['bert-base-uncased','bert-large-uncased']
        self.glove_dict = ['glove.6B.50d','glove.6B.100d','glove.6B.200d','glove.6B.300d','glove.840B.300d']
        self.character_dict = ['glove.840B.300d-char']
        self.embed_dim = dataset_config.embed_dim
        self.pretrained = dataset_config.pretrained
        self.embedding_type = self.pretrained_type()
        self.model_idx = dataset_config.model_idx
        
    def load_embedding(self, dictionary):
        if self.embedding_type in ['char', 'word']:
            embedding_matrix = None
        elif self.embedding_type in ['glove_char', 'glove_word']:
            embedding_matrix = self.get_embedding_matrix(dictionary)
        elif self.embedding_type in ['transformer_word']:
            embedding_matrix = self.pretrained
        else:
           raise ValueError(f'Unknown pretrained mode, pretrained = [{self.pretrained}], embedding_type = [{self.embedding_type}].')
        return embedding_matrix

    def avg_len(self, matrices):
        ret = []
        for matrix in matrices:
            tmp_len = [len(item) for item in matrix]
            ret.append(sum(tmp_len) / len(tmp_len))
        return ret

    def target_statisic(self, targets):
        ret = []
        for target_vec in targets:
            tmp_target = {}
            for item in target_vec:
                if not item in tmp_target:
                    tmp_target[item] = 0
                tmp_target[item] += 1
            ret.append(tmp_target)
        return ret

    def pretrained_type(self):
        if self.pretrained == 'word':
            ret = 'word'
        elif self.pretrained == 'char':
            ret = 'char'
        elif self.pretrained in self.glove_dict:
            ret = 'glove_word' 
        elif self.pretrained in self.transformers_dict:
            ret = 'transformer_word' 
        elif self.pretrained in self.character_dict:
            ret = 'glove_char'
        else:
            raise ValueError(f'Unsupported pretrained mode [{self.pretrained}]')
        return ret

    def tokenize(self, line, spliter = ' ', pre_eos = False, end_eos = True, lower = True, remove_punc = False, character = False):
        if lower: line = line.lower()
        if remove_punc: line =  ''.join(c for c in line if c not in string.punctuation)
        line = re.compile("\s+").sub(" ", line) # remove extra spaces ('a  b  c' -> 'a b c')
        line = line.strip()

        words = list(line) if character else line.strip().split(spliter)

        if pre_eos: words = ['<eos>'] + words
        if end_eos: words.append('<eoe>')

        return words

    def get_embedding_matrix(self, dictionary):
        pretrained = "datasets/glove.6B/" + self.pretrained + '.txt'
        missing = 0
        glove_dict = {}
        emb_weight = np.random.random([len(dictionary), self.embed_dim]).astype(np.float32)
        with open(pretrained, 'r', encoding = 'utf-8') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                line = line.strip().split(' ')
                word = line[0]
                embedding = np.array(line[1:], dtype = np.float32)#[float(x) for x in line[1:]]
                glove_dict[word] = embedding
                # lines.set_description(f"Loaing pretrained: {word.ljust(16,' ')}")  
                lines.set_description(f"Loaing pretrained...")  
                
        for idx, (key, v) in enumerate(dictionary.word2idx.items()):
            if key in glove_dict:
                emb_weight[idx] = glove_dict[key]
            else:
                missing += 1
        print(f'Dictionary has been built, missing token = [{missing}]')
        return emb_weight, missing

class Dictionary_def(object):
    def __init__(self, special = True):
        self.word2idx = {} # word -> index
        self.idx2word = [] # index -> word
        self.counter = Counter()

        self.pad = '<pad>'
        self.eos = '<unk>'
        self.eoe = '<eos>'
        self.unk = '<eoe>'

        self.add_word(self.pad)
        self.add_word(self.eos)
        self.add_word(self.eoe)
        self.add_word(self.unk)

        self.nspecial = len(self.idx2word)

    def add_word(self, word, freq = 1): # 检查这个函数并从这里开始继续
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        token_id = self.word2idx[word]
        self.counter[word] += freq
        return token_id
    
    def get_index(self, word, unk = False):
        if word in self.word2idx:
            return self.word2idx[word]
        if unk:
            return self.word2idx[self.unk]
        raise RuntimeError('Unknown word [{}], and return unk = False'.format(word))

    def __len__(self):
        assert len(self.idx2word) == len(self.word2idx), 'Unexpected error.'
        return len(self.idx2word)

    def build_char_dict(self, upper = False):
        charbox = list('abcdefghijklmnopqrstuvwxyz 0123456789!@#$%^&*()_+-=[]\\;\',./:"<>?|}{~`')
        if upper: charbox = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + charbox
        for c in charbox:
            self.add_word(c)

class Dictionary_emp(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.counter = Counter()
        self.nspecial = 0

    def add_word(self, word, freq = 1): # 检查这个函数并从这里开始继续
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        token_id = self.word2idx[word]
        self.counter[word] += freq
        return token_id
    
    def get_index(self, word, unk = False):
        if word in self.word2idx:
            return self.word2idx[word]
        if unk:
            return self.word2idx[self.unk]
        raise RuntimeError('Unknown word [{}], and return unk = False'.format(word))

    def __len__(self):
        assert len(self.idx2word) == len(self.word2idx), 'Unexpected error.'
        return len(self.idx2word)
    
    def build_char_dict(self):
        charbox = list('abcdefghijklmnopqrstuvwxyz 0123456789!@#$%^&*()_+-=[]\\;\',./:"<>?|}{~`')
        for c in charbox:
            self.add_word(c)

# ---------------------------------------------------------
def load_dataset(data_name):
    return dataset_dict[data_name].setup_dataset()

def register_dataset(name):
    def register_dataset(cls):
        if name in dataset_dict:
            raise ValueError('Cannot register duplicate dataset ({})'.format(name))
        dataset_dict[name] = cls
        return cls
    return register_dataset

for dirname in dataset_names:
    for file in os.listdir(os.path.dirname(__file__ ) + '/' + dirname):
        if file.endswith('.py') and not file.startswith('_'):
            module = file[:file.find('.py')]
            importlib.import_module('program.datasets.{}.'.format(dirname) + module)






    






















