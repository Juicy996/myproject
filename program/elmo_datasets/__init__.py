import os, sys, time, random, importlib, torch, string, re
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from ..datasets import datasets_utils

dataset_dict = {}
dataset_names = ['datasets_translation', 'datasets_classification', 'datasets_netgram', 'datasets_multi']

class FeatureExtractor():
    def __init__(self, selfelmo_weight = None, elmo_option = None):
        '''Only work in CPU'''
        super().__init__()
        pre = 'datasets/elmo_pretrained/'
        weights_file = pre + 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        options_file = pre + 'option.json'
        self.encoder = Elmo(options_file, weights_file, 1, dropout = 0)
        if torch.cuda.is_available(): self.encoder = self.encoder.cuda()
        self.encoder.eval()
        print(f"Elmo initialized with options:\n{options_file}\n{weights_file}.", end='\n\n')

    def __call__(self, tokens):
        """Elmo representation is extracted for each candidate in a turn."""
        token_ids = batch_to_ids(tokens)
        if torch.cuda.is_available(): token_ids = token_ids.cuda()
        embeddings = self.encoder(token_ids)["elmo_representations"][0].detach().cpu().data
        return embeddings

class basic_dataset():
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.elmo = FeatureExtractor()

    def basic_load_dataset(self, file_trn: str, file_val: str, file_tst: str): # 返回划分好的数据集 np(np)
        trn = self.load_file(file_trn)
        val = self.load_file(file_val) if file_val is not None else [None, None, None]
        tst = self.load_file(file_tst) if file_tst is not None else [None, None, None]

        source, wizard, target = self.permute(trn, val, tst)

        source_trn  = np.array(source[:-(self.val_num + self.tst_num)], dtype = object)
        wizard_trn  = np.array(wizard[:-(self.val_num + self.tst_num)], dtype = object) if wizard is not None else None
        target_trn  = torch.tensor(target[:-(self.val_num + self.tst_num)])

        source_val  = np.array(source[-(self.val_num + self.tst_num) : -self.tst_num], dtype = object)
        wizard_val  = np.array(wizard[-(self.val_num + self.tst_num) : -self.tst_num], dtype = object) if wizard is not None else None
        target_val  = torch.tensor(target[-(self.val_num + self.tst_num) : -self.tst_num])

        source_tst  = np.array(source[-self.tst_num:], dtype = object)
        wizard_tst  = np.array(wizard[-self.tst_num:], dtype = object) if wizard is not None else None
        target_tst  = torch.tensor(target[-self.tst_num:])
        print(f'Train num = {len(target_trn)}, valid num = {len(target_val)}, test num = {len(target_tst)}')
        
        return source_trn, wizard_trn, target_trn, source_val, wizard_val, target_val, source_tst, wizard_tst, target_tst

    def tokenize(self, 
                 line,
                 spliter = ' ', 
                 pre_eos = False, 
                 end_eos = True, 
                 lower = True, 
                 remove_punc = False,
                 stopwords = None,
                ):
        stops = []
        if stopwords:
            assert os.path.exists(stopwords), f'Cannot load stopwords [{stopwords}], file not found.'
            with open(stopwords) as f:
                stops = [stopw for stopw in f.readlines()]

        if lower: line = line.lower()
        if remove_punc: line = self.remove_punctuations(line)
        line = re.compile("\s+").sub(" ", line) # remove extra spaces ('a  b  c' -> 'a b c')
        line = line.strip()
        words = [tmpw for tmpw in line.strip().split(spliter) if not tmpw in stops]

        if pre_eos: words = ['<eos>'] + words
        if end_eos: words.append('<eoe>')
        words = self.elmo([words])
        return words

    def remove_punctuations(self, text):
        b = ''.join(c for c in text if c not in string.punctuation)
        return b

    def batchify(self, tvt, batch_size, pad_mode = 'post', shuffle = True, same_len = True, seq_len = None):
        if tvt == 'trn':
            source_tmp = self.source_trn
            wizard_tmp = self.wizard_trn
            target_tmp = self.target_trn
        elif tvt == 'val':
            source_tmp = self.source_val
            wizard_tmp = self.wizard_val
            target_tmp = self.target_val
        elif tvt == 'tst':
            source_tmp = self.source_tst
            wizard_tmp = self.wizard_tst
            target_tmp = self.target_tst
        else:
            raise ValueError('tvt value must in [trn, val, txt].')

        if seq_len is None: # not netgram
            assert len(source_tmp) == len(target_tmp), f'The length of data and label must be same [{len(source_tmp)} and {len(target_tmp)}].'
            assert pad_mode in ['pre', 'post'], 'pad_mode must in [pre, post], got [{}]'.format(pad_mode)

            if shuffle == True:
                source_tmp, wizard_tmp, target_tmp = datasets_utils.shuffle(source_tmp, wizard_tmp, target_tmp)

            nbatch = len(target_tmp) // batch_size
            if not len(target_tmp) % batch_size == 0: nbatch += 1
        else: # netgram
            len(data) // (self.seq_len * batch_size)
            data_tmp = source_tmp
            source_tmp = data_tmp[:num_batch * batch_size * seq_len].reshape([-1, seq_len])
            target_tmp = data_tmp[1: num_batch * batch_size * seq_len + 1].reshape([-1, seq_len])
            same_len = False

        ret = []
        idx = 0
        
        for iterator in range(nbatch):
            if idx + batch_size < len(target_tmp):
                stmp = source_tmp[idx:idx + batch_size]
                wtmp = wizard_tmp[idx:idx + batch_size] if wizard_tmp is not None else None
                ttmp = target_tmp[idx:idx + batch_size]
            elif same_len:
                stmp = source_tmp[-batch_size:]
                wtmp = wizard_tmp[-batch_size:] if wizard_tmp is not None else None
                ttmp = target_tmp[-batch_size:]
            else:
                stmp = source_tmp[idx:idx + batch_size]
                wtmp = wizard_tmp[idx:idx + batch_size] if wizard_tmp is not None else None
                ttmp = target_tmp[idx:idx + batch_size]
            idx += batch_size

            if seq_len is None: # not netgram
                stmp = self.pad_sequence(stmp)
                wtmp = self.pad_sequence(wtmp) if wtmp else None
                # ttmp = torch.from_numpy(ttmp)
            else: # netgram
                stmp = torch.from_numpy(stmp)
                wtmp = torch.from_numpy(wtmp) if wtmp is not None else None
                ttmp = torch.from_numpy(ttmp)


            ret.append([stmp, wtmp, ttmp])
        return ret

    def pad_sequence(self, vecs, pad_value = 0):
        max_len = max([tmp.shape[1] if tmp is not None else 0 for tmp in vecs])
        ret = torch.ones([len(vecs), max_len, 256], dtype = torch.float) * pad_value

        for idx, vec in enumerate(vecs):
            t_vec = torch.tensor(vec)
            ret[idx, :vec.shape[1], :] = t_vec

        return ret.contiguous()

    def print_self(self):
        trn_len = [item.shape[1] for item in self.source_trn]
        val_len = [item.shape[1] for item in self.source_val]
        tst_len = [item.shape[1] for item in self.source_tst]
        print(f'Average length of training source = {sum(trn_len) / len(trn_len)}')
        print(f'Average length of validation source = {sum(val_len) / len(val_len)}')
        print(f'Average length of test source = {sum(tst_len) / len(tst_len)}')

        if self.wizard_trn is not None:
            trn_len = [item.shape[1] for item in self.wizard_trn]
            val_len = [item.shape[1] for item in self.wizard_val]
            tst_len = [item.shape[1] for item in self.wizard_tst]
            print(f'Average length of training wizard = {sum(trn_len) / len(trn_len)}')
            print(f'Average length of validation wizard = {sum(val_len) / len(val_len)}')
            print(f'Average length of test wizard = {sum(tst_len) / len(tst_len)}')
        else:
            print('Wizard is None')

        if self.task_type in ['c']:
            trn_target = {}
            for tmp in self.target_trn:
                item = tmp.item()
                if not item in trn_target:
                    trn_target[item] = 0
                trn_target[item] += 1
            
            val_target = {}
            for tmp in self.target_val:
                item = tmp.item()
                if not item in val_target:
                    val_target[item] = 0
                val_target[item] += 1
            
            tst_target = {}
            for tmp in self.target_tst:
                item = tmp.item()
                if not item in tst_target:
                    tst_target[item] = 0
                tst_target[item] += 1
            
            print(f'Training targets = {trn_target}')
            print(f'Validation targets = {val_target}')
            print(f'Test targets = {tst_target}')
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
            importlib.import_module('program.elmo_datasets.{}.'.format(dirname) + module)






    






















