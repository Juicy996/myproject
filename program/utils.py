import os, hashlib, torch, json, warnings
from . import datasets#, elmo_datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from program.datasets import datasets_utils
from torch.utils.data import DataLoader
from collections.abc import Iterable
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

def try_cuda(cuda, instance):
    if cuda == 0 or not torch.cuda.is_available(): return cuda, instance
    if isinstance(cuda, int):
        mode = [i for i in range(min(cuda, torch.cuda.device_count()))]
    elif isinstance(cuda, str):
        mode = int(cuda)
    else:
        raise ValueError(f'Unsupported cuda mode {cuda}')
    if isinstance(mode, list) and len(mode) > 1:
        instance = torch.nn.DataParallel(instance, device_ids = mode)
        return mode, instance.cuda()
    elif isinstance(mode, list) and len(mode) == 1:
        mode = mode[0]
    instance = instance.to(mode)
    return mode, instance

def set_freeze_all(model, freeze = True):
    for param in model.parameters():
        param.requires_grad = not freeze

    for name, child in model.named_children():
        for param in child.parameters():
            #print(param.name)
            param.requires_grad = not freeze

def set_freeze_by_names(model, layer_names, freeze = True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            #print(param.name)
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze = True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)
 
def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def heatmap(matrix, cmap = 'Greens'):
    # vmin = 0, vmax = 1 取值范围 , center = 0 中心
    sns.set()
    ax = sns.heatmap(matrix, cmap = cmap)
    plt.show()

def save_file(corpus, save_path):
    # Load / save dataset
    torch.save(corpus, save_path)
    print('Dataset [{}] was saved at [{}].'.format(corpus._name, save_path))

def load_file(dataset_name, save_path):
    ret = torch.load(save_path)
    return ret

def get_save_path(dataset_name, dataset_config):
    config_list = [dataset_name]
    for k in ['embed_dim', 'pretrained', 'nvocab_src', 'nvocab_tgt', 'nvocab', 'entrance']:
        tmp = dataset_config.__dict__[k]
        config_list.append(tmp)
    config_text = '_'.join([str(tmp) for tmp in config_list])
    ret = f'datasets/saved/{dataset_name}_{config_text}.dataset'
    return ret

def load_dataset(dataset_name, dataset_config):
    data_path = get_save_path(dataset_name, dataset_config)
    if os.path.exists(data_path) and not dataset_config.force:
        print(f'Loading saved dataset [{dataset_name}] from [{data_path}]')
        ret = load_file(dataset_name, data_path)
    else:
        if dataset_config.force: print(f'Forced to build dataset [{dataset_name}]')
        else: print('Dataset [{}] hasn\'t preprossed yet, Building from raw data.'.format(dataset_name))
        ret = datasets.load_dataset(dataset_name)(dataset_config)
        save_file(ret, data_path)

    if ret.task_type in ['n']:
        tmp_bsz = dataset_config.batch_size
        seq_len = dataset_config.seq_len
        ret.loader = {
            'trn': DataLoader(ret.trn.equip(tmp_bsz, seq_len), batch_size = tmp_bsz, shuffle = False),
            'val': DataLoader(ret.val.equip(tmp_bsz, seq_len), batch_size = tmp_bsz, shuffle = False),
            'tst': DataLoader(ret.tst.equip(tmp_bsz, seq_len), batch_size = tmp_bsz, shuffle = False),
        }
    else:
        ret.loader = {
                'trn': DataLoader(ret.trn, batch_size = dataset_config.batch_size, collate_fn = datasets_utils.pad_sequence, shuffle = ret.shuffle),
                'val': DataLoader(ret.val, batch_size = dataset_config.batch_size, collate_fn = datasets_utils.pad_sequence, shuffle = False),
                'tst': DataLoader(ret.tst, batch_size = dataset_config.batch_size, collate_fn = datasets_utils.pad_sequence, shuffle = False),
            }
    ret.print_self()
    return ret

def adjust_learning_rate(optimizer, cur_epoch, max_epoch, init_lr):
    new_lr = 0.8 ** cur_epoch * init_lr
    set_learning_rate(optimizer, new_lr, max_epoch = max_epoch, init_lr = init_lr)
    # print('New LR = {}'.format(lr))

def set_learning_rate(optimizer, lr, verbose = True, max_epoch = 0, init_lr = 0):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if verbose and max_epoch == 0: print(f'New LR = [{lr}] has beed set.')
    if verbose: print(f'New LR = [{lr}] has beed set, min_LR = [{0.8 ** max_epoch * init_lr}].')

def manage_extra_output(dict_output):
    if dict_output is None or len(dict_output) == 0:
        return None
    if isinstance(dict_output, dict):
        for k, v in dict_output.items():
            dict_output[k] = manage_extra_output(v)
        return dict_output
    elif isinstance(dict_output, torch.Tensor):
        return dict_output.detach()
    else:
        return tuple(repackage_hidden(v) for v in dict_output)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def plot_roc(logit_path):
    nploaded = np.loadtxt(logit_path)
    logits = nploaded[:, 1:]
    target = nploaded[:, 0].squeeze().astype(np.int32)
    n_classes = logits.shape[1]
    target = np.eye(n_classes)[target]
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(["#51C1C8", "#E96279", "#44A2D6", "#536D84", '#800080', '#00994e', '#ff6600', 'aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Yelp SLSTM')
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig('C:/Users/ap/Desktop/nc/fig/rocs/figname.png', dpi = 600)

class Getch:
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()

class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

class ResultRecoder(object):
    def __init__(self):
        self.cur_epoch = 0
        self.best_val_loss = 9999999999999.0
        self.best_val_eval = -1.0
        self.output_loss = None
        self.output_eval = None
        self.cur_lepoch = -1
        self.cur_eepoch = -1

        self.global_best_eval = -1
        self.global_best_loss = 99999999999

    def push_loss(self, cur_epoch, undec, val_loss, ret_tst):
        if val_loss < self.best_val_loss:
            print('New val loss [{}] < previous loss [{}]'.format(val_loss, self.best_val_loss))
            self.cur_lepoch = cur_epoch
            self.best_val_loss = val_loss
            self.output_loss = ret_tst
            undec = 0
            print('New output results ')
            self.print_output(self.output_loss)
            print(f'detected at epoch [{self.cur_lepoch}]')
            return undec
        else:
            print('New val loss [{}] > previous loss [{}], results unchanged'.format(val_loss, self.best_val_loss))
            self.print_output(self.output_loss)
            print(f'at epoch [{self.cur_lepoch}]')
            return undec + 1

    def push_eval(self, cur_epoch, undec, val_eval, ret_tst):
        if val_eval > self.best_val_eval:
            print('New val eval [{}] > previous eval [{}]'.format(val_eval, self.best_val_eval))
            self.cur_eepoch = cur_epoch
            self.best_val_eval = val_eval
            self.output_eval = ret_tst
            undec = 0
            print('New output results ')
            self.print_output(self.output_eval)
            print(f'detected at epoch [{self.cur_eepoch}]')
            return undec
        else:
            print('New val eval [{}] < previous eval [{}], results unchanged.'.format(val_eval, self.best_val_eval))
            self.print_output(self.output_eval)
            print(f'at epoch [{self.cur_eepoch}]')
            return undec + 1
        
    def pop_via_loss(self):
        ret_tmp = None
        for k, v in self.output_eval.items():
            if 'loss' in k:
                ret_tmp = v
        return self.cur_lepoch, ret_tmp
    
    def pop_via_eval(self):
        ret_tmp = None
        for k, v in self.output_eval.items():
            if 'acc' in k:
                ret_tmp = v
        return self.cur_eepoch, ret_tmp

    def print_output(self, output_dict):
        ret_loss = 0
        ret_eval = 0
        for k, v in output_dict.items():
            if 'loss' in k: ret_loss += v
            if 'acc' in k: ret_eval += v

        for k, v in output_dict.items(): print(f'{k} = {v}')
        print(f'total_loss = [{ret_loss}]')
        print(f'total_eval = [{ret_eval}]')

class statistic(object):
    def __init__(self):
        self.record = {}
        self.ret = {}
    
    def push(self, task_ret):
        for k, v in task_ret.items():
            if k == 'loss' or k == 'extra_output' or k == 'rep': continue
            if not isinstance(v, torch.Tensor):
                warnings.warn(f"{k} is not a tensor.")
                continue

            if not k in self.record: 
                self.record[k] = []

            # self.record[k].append(v.unsqueeze(0) if len(v.shape) == 0 else v)
            self.record[k].append(v)

    def pop(self):
        ret = {}
        for k, v in self.record.items():
            tmp = torch.cat(v)
            ret[k] = torch.mean(tmp) if 'loss' in k else torch.sum(tmp)
            self.record[k] = []
        return ret

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class LoggerBoard(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


















