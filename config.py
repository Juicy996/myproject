import os, sys, torch, json, argparse

def load_params():
    parser = argparse.ArgumentParser()
    pa = parser.add_argument
    #---str
    pa("-r", "--resume", type = str, default = 'None', help = "Saved model path")
    pa("-s", "--seed", type = int, default = -1, help = "Random seed")
    pa("-c", "--config_file", type = str, default = 'automl', help = "config_file")

    args = parser.parse_args()
    return args

class hyper_params():
    def __init__(self, args, dict_config):
        self.apex = True, # 使用NVIDIA-APEX
        self.backward_interval = 1 # 每隔多少个batch梯度更新
        self.clip = 5.0 # 梯度裁剪
        self.cuda = 2 # GPU数量
        self.embed_dim = 300 # embed 维数
        self.evaluation = 'loss'  # 优化目标 [loss, acc]
        self.hidden_dim = 300 # 模型维数
        self.log_file = False # 日志存储位置 True: auto path, False: None, str: specificed path
        self.log_interval = 1 # 输出日志间隔
        self.log_print = True # 是否输出到屏幕
        self.lr = 0.0001 # 初始学习率，所有学习率衰减函数都根据初始学习率计算
        self.lr_update = True # 学习率是否更新
        self.max_epoch = 20 # 最大训练轮数
        self.min_trust = 0.25 # lamb优化器的参数
        self.optimizer = 'adam' # 优化器
        self.same_num_batch = False # 重复训练较小的数据集，让不同数据集具有相同的batch -- 暂未实现
        self.stop_val_dec = 5 # 提前结束训练
        self.template_count = 10 # template 第2维，第1维固定为1
        self.tensorlog_path = None # tensorlog存储位置
        self.tksize = [5, 5] # template 卷积核大小 (二维)
        self.val_interval = 1 # 每隔多少个epoch运行验证集
        self.wait_for_synchronization = True # 同步不同大小的数据集的batch -- 暂未实现
        self.warmup = 0 # 学习率预热
        self.weight_decay = 0.0 #权重衰减 -- 暂未实现
        self.error_analysis = False
        self.nni = True
        self.save_model = False
        for k, v in dict_config.items():
            if hasattr(self, k): setattr(self, k, v)
        for k, v in args.__dict__.items():
            self.k = v

class template_config():
    def __init__(self, hp, dict_config):
        self.template_count = hp.template_count # template 第2维，第1维固定为1
        self.template_initializer = 'xavier_uniform_' # 初始化函数
        self.nrow = 304
        self.ncol = 304

        for k, v in dict_config.items():
            if hasattr(self, k): setattr(self, k, v)

class dataset_config():
    def __init__(self, hp, k, dict_config):
        self.dim_dict = {
            'bert-base-uncased': 768,
            'gpt2': 768,
            'gpt2-medium': 1024,
            'gpt2-large': 1280,
            'gpt2-xl': 768,
            'bert-large-uncased': 1024,
            'glove.6B.300d': 300,
            'glove.840B.300d-char': 300,
            'glove.6B.100d': 100,
            'glove.6B.50d': 50,

        }
        self.batch_size = 20 # batch_size
        self.dataset = None # dataset
        self.embed_dim = hp.embed_dim # embed 维数
        self.embedding_trainable = True # embedding是否可以训练 (pretrained 不等于 None的时候可用)
        self.entrance = 'main'
        self._name = k # 名称
        self.force = False # 必须从文件读取数据集
        self.model_idx = [0] # 使用模型的序号
        self.model_weight = [1] # 每个model所占权重
        self.nvocab = 99999999999999 # 词典最大长度
        self.nvocab_src = 99999999999999 # 词典最大长度
        self.nvocab_tgt = 99999999999999 # 词典最大长度
        self.pos_embed = False # pos_embed -- 暂未实现
        self.pretrained = None # glove等文件位置
        self.seq_len = None # 序列长度(netgram模型可用)
        self.weight = 1.0 # 对val loss(evla)的影响权重
        
        for k, v in dict_config.items():
            if hasattr(self, k): setattr(self, k, v)
        
        if self.pretrained is None: raise ValueError(f'Attribute [pretrained] must be specificed')
        elif self.pretrained in ['word', 'char']:
            pass
        elif self.pretrained in self.dim_dict:
            print(f'Embed_dim was set from [{self.embed_dim}] to [{self.dim_dict[self.pretrained]}] to fit [{self.pretrained}].')
            self.embed_dim = self.dim_dict[self.pretrained]
        else:
            raise ValueError(f'Unknown attribute pretrained = [{self.pretrained}]')

class model_config():
    def __init__(self, hp, dict_config):
        self.embed_dim = hp.embed_dim # embed 维数
        self.hidden_dim = hp.hidden_dim # 模型维数
        self.model = None # model
        self.model_name = dict_config['model_name'] # 模型名称
        self.nlayer = 1 # 层数
        self.nhead = 1 # 头数
        self.cache_len = 1 # 历史信息数
        self.dropp = 0.1 # dropout

        for k, v in dict_config.items():
            if hasattr(self, k): setattr(self, k, v)

class criterion_config(): 
    def __init__(self, hp, dict_config):
        self.hidden_dim = hp.hidden_dim # 模型维数
        self.criterion = None # criterion
        self.criterion_name = dict_config['criterion_name'] # loss名称
        self.num_label = None # 标签数量
        self.cutoffs = [] # 分段统计 -- 暂未实现
        self.div_val = 1 # 暂未实现
        self.keep_order = False # 暂未实现
        self.lbd = 0.5 # lambda for soe

        for k, v in dict_config.items():
            if hasattr(self, k): setattr(self, k, v)

class optimizer_config():
    def __init__(self, hp):
        self.lr = hp.lr # 学习率初值
        self.optimizer_name = hp.optimizer # 优化器名称
        self.clip = hp.clip # 梯度裁剪
        self.weight_decay = hp.weight_decay # 权重衰减 -- 暂未实现
        self.lr_update = hp.lr_update # 学习率更新
        self.min_trust = hp.min_trust # lamb 参数

def load_configs(args, file_name):
    with open(file_name, 'r') as f:
        config_dict = json.load(f)
        # fp = open(file_name, 'r')
        # config_dict = json.load(fp)
        print('------------- config --------------')
        print(json.dumps(config_dict, indent = 2))
        print('-----------------------------------')
        hp = hyper_params(args, config_dict['hyper_params'])
        oc = optimizer_config(hp) # 只支持一个optimizer模式

        # template_config, 在多卡环境下不能使用parameterlist
        tcs = {'is_none': config_dict['template_config_set']['is_none']}
        for k, v in config_dict['template_config_set'].items():
            if k == 'is_none': continue
            tcs[k] = template_config(hp, v)

        dcs = {}
        for k, v in config_dict['dataset_config_set'].items():
            dcs[k] = dataset_config(hp, k, v)

        mcs = {}
        for k, v in config_dict['model_config_set'].items():
            mcs[k] = model_config(hp, v)
        
        ccs = {}
        for k, v in config_dict['criterion_config_set'].items():
            ccs[k] = criterion_config(hp, v)
        
        return hp, tcs, dcs, mcs, ccs, oc
        























