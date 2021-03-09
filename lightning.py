import os,torch, argparse, random
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from program import datasets, models, criterions, modules, utils

class config():
    def __init__(self):
        # --- hp, optimizer
        self.cuda = 0
        self.clip = 5.0
        self.lr = 0.0001
        self.optimizer_name = 'adam'
        self.lr_update = True
        self.max_epoch = 40
        self.warmup = 800
        self.weight_decay = 0.0
        self.nni = False
        self.seed = -1
        # --- dataset
        self.dataset_name = 'mr_16' # 名称
        self.batch_size = 20 # batch_size
        self.embed_dim = 300 # embed 维数
        self.embedding_trainable = False # embedding是否可以训练 (pretrained 不等于 None的时候可用)
        self.force = False # 必须从文件读取数据集
        self.model_idx = [0] # deprecated 
        self.model_weight = [1] # deprecated 
        self.nvocab = 99999999999999 # 词典最大长度
        self.nvocab_src = 99999999999999 # 词典最大长度
        self.nvocab_tgt = 99999999999999 # 词典最大长度
        self.pos_embed = False # pos_embed -- 暂未实现
        self.pretrained = "glove.6B.300d" # glove等文件位置 与 elmo，character冲突
        self.seq_len = None # deprecated 
        self.weight = 1.0 # deprecated 
        self.elmo = False
        self.character = True
        # --- model
        self.model_name = 'mindicator' # 模型名称
        self.hidden_dim = 300 # 模型维数
        self.nlayer = 1 # 层数
        self.nhead = 1 # 头数
        self.dropp = 0.1 # dropout
        # --- criterion
        self.criterion = None # criterion
        self.criterion_name = 'classification' # loss名称
        self.num_label = None # 标签数量
        # --- optimizer

def load_params():
    hparams = config()

    parser = argparse.ArgumentParser()
    pa = parser.add_argument
    #---str
    pa("-s", "--seed", type = int, default = 1, help = "Random seed")
    pa("-c", "--config_file", type = str, default = 'automl', help = "config_file")
    pa("--batch_size", type = int, default = 5, help = "batch_size")
    pa("--dataset_name", type = str, default = None, help = "dataset_name")
    pa("--max_epoch", type = int, default = 40, help = "max_epoch")

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        if k in hparams.__dict__ and getattr(args, k) is not None:
            setattr(hparams, k, v)
    return hparams

def seed_torch(seed = 1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    print(f'Random seed was set with {seed}.')

def create_embeddings(config, dataset):
    ret = None
    if dataset.task_type == 'n':
        raise ValueError('Do not support netgram.')

    if config.elmo: ret = None

    elif dataset.task_type in ['c', 'd']:
        ret = modules.Embedding(ntoken = len(dataset.dict), 
                                embed_dim = config.embed_dim, 
                                pretrained_matrix = dataset.embedding_matrix, 
                                trainable = config.embedding_trainable,
                                )

    elif dataset.task_type in ['t', 'm']:
        emb_src = modules.Embedding(ntoken = len(dataset.dict[0]), 
                                    embed_dim = config.embed_dim, 
                                    pretrained_matrix = dataset.embedding_matrix[0], 
                                    trainable = config.embedding_trainable,
                                    )
        emb_tgt = modules.Embedding(ntoken = len(dataset.dict[1]), 
                                    embed_dim = config.embed_dim, 
                                    pretrained_matrix = dataset.embedding_matrix[1], 
                                    trainable = config.embedding_trainable,
                                    )
        ret = nn.ModuleDict({'emb_src': emb_src, 'emb_tgt': emb_tgt})

    else:
        warnings.warn(f'Unknown dataset_type [{dataset.task_type}], embedding will be set with None.')
        ret = None

    return ret

class Task(pl.LightningModule):
    def __init__(self, hp, dataset):
        super().__init__()
        self.hparams = hp.__dict__
        self.net = models.load_model(hp.model_name)(hp, None)
        self.criterion = criterions.load_criterion(hp.criterion_name)(64, 10)
        self.embedding = create_embeddings(hp, dataset)
    
    def training_step(self, batch, batch_idx):
        rep, _ = self.net(batch, self.embedding, None, None, None)
        net_ret = self.criterion(rep, batch[-1], extra_input = None)

        loss = net_ret['loss']
        corrects = net_ret['correct']
        total = net_ret['total']

        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
        self.log('trn_loss', loss)
        # self.log('trn_correct', corrects, reduce_fx = torch.sum)
        # self.log('trn_total', total, reduce_fx = torch.sum)
        self.log('trn_acc', corrects / total)

        return loss

    def validation_step(self, batch, batch_idx):
        rep, _ = self.net(batch, self.embedding, None, None, None)
        net_ret = self.criterion(rep, batch[-1], extra_input = None)

        loss = net_ret['loss_detach']
        corrects = net_ret['correct']
        total = net_ret['total']

        self.log('val_loss', loss)
        # self.log('val_correct', corrects, reduce_fx = torch.sum)
        # self.log('val_total', total, reduce_fx = torch.sum)
        self.log('val_acc', corrects / total)

        return loss

    def test_step(self, batch, batch_idx):
        rep, _ = self.net(batch, self.embedding, None, None, None)
        net_ret = self.criterion(rep, batch[-1], extra_input = None)

        loss = net_ret['loss_detach']
        corrects = net_ret['correct']
        total = net_ret['total']

        self.log('tst_loss', loss)
        # self.log('tst_correct', corrects, reduce_fx = torch.sum)
        # self.log('tst_total', total, reduce_fx = torch.sum)
        self.log('tst_acc', corrects / total)

        return loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.parameters(), lr=0.01)
        # dis_opt = torch.optim.AdamAdam(self.model_disc.parameters(), lr=0.02)
        gen_sched = torch.optim.lr_scheduler.ExponentialLR(gen_opt, 0.99)  # called after each training step
        # dis_sched = torch.optim.lr_scheduler.CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
        return [gen_opt], [gen_sched]
    
class tvt_dataset(torch.utils.data.Dataset):
    def __init__(self, source, wizard, target):
        super().__init__()
        self.source = source
        self.wizard = wizard
        self.target = target

    def __getitem__(self, index):
        src = self.source[index]
        wiz = self.wizard[index] if self.wizard else torch.tensor(0)
        tgt = self.target[index]
        return [src, wiz, tgt]

    def __len__(self):
        return len(self.target)

def pad_sequence(samples, pad_mode = 'post', pad_value = 0):
    sources = []
    wizards = []
    targets = []
    for sample in samples:
        sources.append(sample[0])
        wizards.append(sample[1])
        targets.append(sample[2])
    
    max_len = max([len(tmp) if tmp is not None else 0 for tmp in sources])
    ret = torch.ones([len(sources), max_len], dtype = torch.int32) * pad_value

    for idx, vec in enumerate(sources):
        t_vec = torch.from_numpy(vec)
        if pad_mode == 'post':
            ret[idx][:len(vec)] = t_vec
        else:
            ret[idx][-len(vec):] = t_vec
    return ret.contiguous(), torch.tensor(wizards), torch.tensor(targets)

if __name__ == '__main__':
    hparams = load_params()
    seed_torch(hparams.seed if hparams.seed > 0 else random.randint(1, 99999999999))

    dataset = utils.load_dataset(hparams)
    hparams.num_label = dataset.num_label

    train = tvt_dataset(dataset.source_trn, dataset.wizard_trn, dataset.target_trn)
    val = tvt_dataset(dataset.source_val, dataset.wizard_val, dataset.target_val)
    tst = tvt_dataset(dataset.source_tst, dataset.wizard_tst, dataset.target_tst)

    trn_loader = DataLoader(train, batch_size = hparams.batch_size, collate_fn = pad_sequence)
    val_loader = DataLoader(val, batch_size = hparams.batch_size, collate_fn = pad_sequence)
    tst_loader = DataLoader(tst, batch_size = hparams.batch_size, collate_fn = pad_sequence)

    model = Task(hparams, dataset)

    logger = TensorBoardLogger('lightning_logs', name='sync')
    trainer = pl.Trainer(
        progress_bar_refresh_rate = 10, 
        logger = logger, 
        max_epochs = hparams.max_epoch,
        auto_lr_find = True,
        # gpus = '0', #hparams.cuda,
        # accelerator = 'ddp',
        gradient_clip_val = hparams.clip,
    )

    trainer.fit(model, trn_loader, val_loader)
    trainer.test(model, tst_loader, ckpt_path = 'None')
    trainer.test(model, tst_loader, ckpt_path = 'best')
    print(f'Random seed = {hparams.seed}.')
























