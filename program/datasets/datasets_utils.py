import os, sys, importlib, string, re, torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class Dictionary_def(object):
    def __init__(self, special = True):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()

        self.pad = '<pad>'
        self.eos = '<unk>'
        self.eoe = '<eos>'
        self.unk = '<eoe>'

        self.add_word(self.pad)
        self.add_word(self.eos)
        self.add_word(self.eoe)
        self.add_word(self.unk)

        self.special = len(self.idx2word)

    def add_word(self, word, freq = 1): # 检查这个函数并从这里开始继续
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        token_id = self.word2idx[word]
        self.counter[word] += freq
        self.total += freq
        return token_id
    
    def get_index(self, word, unk = False):
        if word in self.word2idx:
            return self.word2idx[word]
        if unk:
            return self.word2idx[self.unk]
        raise RuntimeError('Unknown word [{}]'.format(word))

    def pop(self, key):
        del self.counter[key]
        self.idx2word.remove(key)
        self.word2idx.pop(key)

    def __len__(self):
        return len(self.idx2word)

class Dictionary_emp(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word, index):
        if word in self.word2idx:
            assert index == self.word2idx[word], f'Word index cannot be changed, got {word} changes from {self.word2idx[word]} to {index}.'
        else:
            self.word2idx[word] = index
            self.idx2word[index] = word
        input(word)
        input(index)
        input(self.word2idx[word])
    
    def add_line(self, words, indices):
        for word, index in zip(words, indices[0]):
            self.add_word(word, index)

    def __len__(self):
        return len(self.idx2word)

class transformer_tokenizer(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(mode)
    def __call__(self, line):
        ret = self.tokenizer(line, return_tensors = "pt")
        ret = ret['input_ids'].squeeze()
        ret = ret[: 512]
        return ret

class transformer_model(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained) 
    
    def forward(self, vec, ret_mode = 0):
        ''' ret_mode = 0: only return embedding,
            ret_mode = 1: only return vector
            ret_mode = 2: return embedding and vector'''
        vec = {'input_ids': vec}
        ret = self.model(**vec)

        if ret_mode == 0:
            return ret[0]
        elif ret_mode == 1:
            return ret[1]
        else:
            return ret

def pad_sequence(samples, pad_mode = 'post', pad_value = 0):
    sources = []
    wizards = []
    targets = []
    for sample in samples:
        sources.append(sample[0])
        wizards.append(sample[1])
        targets.append(sample[2])
    
    max_len_src = max([len(tmp) for tmp in sources])
    ret_src = np.ones([len(sources), max_len_src], dtype = sources[0][0].dtype) * pad_value

    for idx, vec in enumerate(sources):
        t_vec = vec
        if pad_mode == 'post':
            ret_src[idx][:len(t_vec)] = t_vec
        else:
            ret_src[idx][-len(t_vec):] = t_vec
    
    max_len_wiz = max([0 if tmp == 0 else len(tmp) for tmp in wizards])
    if max_len_wiz == 0:
        wizards = None
    else:
        ret_wiz= np.ones([len(wizards), max_len_wiz], dtype = wizards[0][0].dtype) * pad_value
        for idx, vec in enumerate(wizards):
            t_vec = vec
            if pad_mode == 'post':
                ret_wiz[idx][:len(t_vec)] = t_vec
            else:
                ret_wiz[idx][-len(t_vec):] = t_vec

    targets = np.array(targets)

    sources = torch.from_numpy(ret_src)
    wizards = None if max_len_wiz == 0 else torch.from_numpy(ret_wiz)
    targets = torch.from_numpy(targets)
    return sources, wizards, targets

def get_embedding_char(word_count, embed_dim):
    emb_weight = np.zeros([word_count, embed_dim], dtype = np.float32)
    for i in range(word_count):
        emb_weight[i, i] = 1.0
    return emb_weight

def shuffle(source, wizard, target):
    indices = np.arange(len(target))
    np.random.shuffle(indices)
    source = source[indices]
    if wizard is not None and len(wizard) > 0: 
        wizard = wizard[indices]
    else:
        wizard = None
    if target is not None and len(target) > 0: 
        target = target[indices]
    else:
        target = None

    return source, wizard, target





















