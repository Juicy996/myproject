import os, sys, time, random
from .. import register_dataset
from . import basic_netgram_dataset

@register_dataset('wikitext103')
class wikitext103(basic_netgram_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_name = 'wikitext103',
                         embed_dim = dataset_config.embed_dim,
                         batch_size = dataset_config.batch_size,
                         pretrained = dataset_config.pretrained,
                         embedding_trainable = dataset_config.embedding_trainable,
                         model_idx = dataset_config.model_idx,
                         seq_len = dataset_config.seq_len,
                         criterion_idx = dataset_config.criterion_idx,
                         encoding = 'utf-8',
        )
        data_dir = 'datasets/datasets_netgram/wikitext103'

        path_trn = os.path.join(data_dir, 'wiki.train.tokens')
        path_val = os.path.join(data_dir, 'wiki.valid.tokens')
        path_tst = os.path.join(data_dir, 'wiki.test.tokens')

        self.load_dataset(path_trn, path_val, path_tst)
        self.num_label = len(self.dict)
        

    @classmethod
    def setup_dataset(cls):
        return cls