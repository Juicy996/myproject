import os, sys, torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .. import datasets_utils
from ... import modules

class basic_translation_dataset():
    def __init__(self, 
                 dataset_name, # name
                 embed_dim, # embed_dim
                 batch_size, # batch_size
                 pretrained, # pretrained glove file
                 embedding_trainable, 
                 model_idx,
                 criterion_idx,
                 encoding,
        ):
        self.task_type = 't'
        self.dataset_name = dataset_name
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.embedding_trainable = embedding_trainable
        self.model_idx = model_idx
        self.criterion_idx = criterion_idx
        self.encoding = encoding
    
    def load_dataset(self, file_trn_src, file_trn_tgt, file_val_src, file_val_tgt, file_tst_src, file_tst_tgt):
        self.dict_src = datasets_utils.build_dict([file_trn_src, file_val_src, file_tst_src], encoding = self.encoding)
        self.dict_tgt = datasets_utils.build_dict([file_trn_tgt, file_val_tgt, file_tst_tgt], encoding = self.encoding)

        self.trn_src = datasets_utils.load_file_unlabel(file_trn_src, self.dict_src, False, self.encoding)
        self.trn_tgt = datasets_utils.load_file_unlabel(file_trn_tgt, self.dict_tgt, False, self.encoding)
        self.val_src = datasets_utils.load_file_unlabel(file_val_src, self.dict_src, False, self.encoding)
        self.val_tgt = datasets_utils.load_file_unlabel(file_val_tgt, self.dict_tgt, False, self.encoding)
        self.tst_src = datasets_utils.load_file_unlabel(file_tst_src, self.dict_src, False, self.encoding)
        self.tst_tgt = datasets_utils.load_file_unlabel(file_tst_tgt, self.dict_tgt, False, self.encoding)

        if self.pretrained is not None:
            self.matrix_src, self.missing_src = datasets_utils.get_embedding_matrix(self.embed_dim, self.dict_src, self.pretrained)
            self.matrix_tgt, self.missing_tgt = datasets_utils.get_embedding_matrix(self.embed_dim, self.dict_tgt, self.pretrained)
        else:
            self.matrix_src, self.missing_src = None, 0
            self.matrix_tgt, self.missing_tgt = None, 0
        print('Dataset [{}] has been built.'.format(self.dataset_name))
    
    def batchify(self, tvt, batch_size, seq_len = None, pad = 'post', shuffle = False, same_len = True):
        if tvt == 'trn':
            src_tmp = self.trn_src
            tgt_tmp = self.trn_tgt
        elif tvt == 'val':
            src_tmp = self.val_src
            tgt_tmp = self.val_tgt
        elif tvt == 'tst':
            src_tmp = self.tst_src
            tgt_tmp = self.tst_tgt
        else:
            raise ValueError('tvt value must in [trn, val, txt].')
        assert len(src_tmp) == len(tgt_tmp), 'The length of source and target must be same.'
        assert pad in ['pre', 'post'], 'pad must in [pre, post]'

        item_count = len(src_tmp)

        if shuffle:
            indices = np.arange(item_count)
            np.random.shuffle(indices)
            source = src_tmp[indices]
            target = tgt_tmp[indices]
        else:
            source = src_tmp
            target = tgt_tmp

        ret = []
        idx = 0

        while idx < item_count:
            if idx + batch_size < item_count:
                src_tmp = source[idx:idx + batch_size]
                tgt_tmp = target[idx:idx + batch_size]
            elif same_len:
                src_tmp = source[-batch_size:]
                tgt_tmp = target[-batch_size:]
            else:
                src_tmp = source[idx:idx + batch_size]
                tgt_tmp = target[idx:idx + batch_size]

            src_tmp = datasets_utils.pad_sequence(src_tmp, pad_mode = pad)
            tgt_tmp = datasets_utils.pad_sequence(tgt_tmp, pad_mode = pad)
            zeros = torch.zeros([src_tmp.shape[0], 1], dtype = torch.int)
            wiz_tmp = torch.cat([zeros, tgt_tmp], dim = 1)
            tgt_tmp = torch.cat([tgt_tmp, zeros], dim = 1)

            ret.append([src_tmp, wiz_tmp, tgt_tmp])

            idx += batch_size
        return ret





























