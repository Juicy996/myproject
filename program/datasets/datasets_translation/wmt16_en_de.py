import os, sys, time, random
from .. import register_dataset
from . import basic_translation_dataset

@register_dataset('wmt16_en_de')
class wmt16_en_de(basic_translation_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_name = 'wmt16_en_de',
                         embed_dim = dataset_config.embed_dim,
                         batch_size = dataset_config.batch_size,
                         pretrained = dataset_config.pretrained,
                         embedding_trainable = dataset_config.embedding_trainable,
                         model_idx = dataset_config.model_idx,
                         criterion_idx = dataset_config.criterion_idx,
                         encoding = 'utf-8',
        )
        data_dir = 'datasets/datasets_translation/wmt16_en_de_google'

        path_trn_src = os.path.join(data_dir, 'train.en')
        path_trn_tgt = os.path.join(data_dir, 'train.de')
        path_val_src = os.path.join(data_dir, 'valid.en')
        path_val_tgt = os.path.join(data_dir, 'valid.de')
        path_tst_src = os.path.join(data_dir, 'test.en')
        path_tst_tgt = os.path.join(data_dir, 'test.de')

        self.load_dataset(path_trn_src, path_trn_tgt, path_val_src, path_val_tgt, path_tst_src, path_tst_tgt, )
        self.num_label = len(self.dict_tgt)
        

    @classmethod
    def setup_dataset(cls):
        return cls