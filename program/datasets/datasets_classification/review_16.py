import os, sys, time, random, re, torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .. import register_dataset, datasets_utils
from . import basic_label_dataset

class review_16(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                indices = tokenizer(line[1:])
                source.append(indices)
                target.append(int(line[0]))
        return source, None, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                words = datasets_utils.tokenize(line[1:], spliter, pre_eos, end_eos, lower, remove_punc)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(int(line[0]))
        return source, None, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        target = []
        with open(filename, 'r', encoding = self.encoding, errors='ignore') as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")

                words = datasets_utils.tokenize(line[1:], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                index = [self.dict.get_index(word, True) for word in words]
                source.append(torch.tensor(index))
                target.append(int(line[0]))
        return source, None, target

    def build_dict_spec(self, file_names):
        dict_ret = datasets_utils.Dictionary_spec()
        # tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        # for file_name in file_names:
        #     if file_name is None: continue
        #     assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

        #     with open(file_name, 'r', encoding = self.encoding) as f:
        #         lines = tqdm(f.readlines(), ascii = True)
        #         for line in lines:
        #             lines.set_description(f"Building dict from {file_name.split('/')[-1]}")
        #             indices = tokenizer(line[1:])
        #             words = 'eos ' + line + 'eoe'
        #             dict_ret.add_line(words, indices)

        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret
    
    def build_dict_index(self, file_names):
        counter = Counter()
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

            with open(file_name, 'r', encoding = self.encoding, errors='ignore') as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {file_name.split('/')[-1]}")

                    words = datasets_utils.tokenize(line[1:], pre_eos = True, end_eos = True)
                    counter.update(words)

        dict_ret = datasets_utils.Dictionary()
        for wid, freq in counter.most_common(self.config.nvocab):
            if not wid in [dict_ret.pad, dict_ret.eos, dict_ret.eoe, dict_ret.unk]:
                dict_ret.add_word(wid, freq)
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    def build_dict_char(self, file_names):
        dict_ret = datasets_utils.build_char_dict()
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

@register_dataset('apparel_16')
class apparel_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/apparel_16'
        self.encoding = 'utf-8-sig'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'apparel.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'apparel.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        # np.array[torch.tensor]

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('baby_16')
class baby_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/baby_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'baby.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'baby.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('books_16')
class books_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/books_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'books.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'books.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('camera_16')
class camera_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/camera_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'camera.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'camera.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('dvd_16')
class dvd_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/dvd_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'dvd.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'dvd.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('electronic_16')
class electronic_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/electronic_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'electronic.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'electronic.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('health_16')
class health_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/health_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'health.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'health.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('imdb_16')
class imdb_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/imdb_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'imdb.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'imdb.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('kitchen_16')
class kitchen_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/kitchen_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'kitchen.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'kitchen.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('magazine_16')
class magazine_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/magazine_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'magazine.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'magazine.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('mr_16')
class mr_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_classification/mr_16'
        self.encoding = 'utf-8-sig'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'MR.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'MR.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('music_16')
class music_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/music_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'music.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'music.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('software_16')
class software_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/software_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'software.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'software.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('sport_16')
class sport_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/sport_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'sport.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'sport.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('toys_16')
class toys_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/toys_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'toys.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'toys.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls

@register_dataset('video_16')
class video_16(review_16):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.encoding = 'utf-8-sig'
        data_dir = 'datasets/datasets_classification/video_16'
        self.num_label = 2

        self.val_num = 160
        self.tst_num = 400

        self.trn_path = os.path.join(data_dir, 'video.task.train')
        self.val_path = None
        self.tst_path = os.path.join(data_dir, 'video.task.test')

        self.load_dataset(self.trn_path, self.val_path, self.tst_path)
        

    @classmethod
    def setup_dataset(cls):
        return cls




























