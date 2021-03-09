import os, sys, time, random, torch, functools, math, json, hashlib, warnings, nni
import numpy as np
import torch.nn as nn
from . import datasets, models, optimizers, task, utils
from tensorboardX import SummaryWriter

try:
    import apex
    from apex import amp
    print('NVIDIA apex is available.')
except:
    warnings.warn("apex is unavailable, failed to import.")

class trainer():
    def __init__(self, hp,
                 template_config,
                 dataset_config_set,
                 model_config,
                 criterion_config,
                 optimizer_config,
                 resume = 'none',
        ):
        # --- config
        self.hp = hp
        self.template_config = template_config
        self.dataset_config_set = dataset_config_set
        self.model_config = model_config
        self.criterion_config = criterion_config
        self.optimizer_config = optimizer_config
        # --- regular config
        self.writer = SummaryWriter(hp.tensorlog_path) if hp.tensorlog_path is not None else None
        self.time_start = time.time()
        self.apex = self.try_apex(hp.apex) if hp.apex else False
        self.cuda = self.hp.cuda
        self.seed = hp.seed
        # --- dynamic config
        self.cur_epoch = 0
        self.recoder = utils.ResultRecoder()
        self.undec = 0
        
        # Example
        # self.writer.add_scalar('quadratic', i**2, global_step=i)
        # self.add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
        # self.add_graph(model, input_to_model=None, verbose=False, **kwargs)
        # self.add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)

        for k, v in self.dataset_config_set.items():
            v.dataset = utils.load_dataset(k, v)
            for mi in v.model_idx:
                self.criterion_config[k + f'_{mi}'].num_label = v.dataset.num_label
             
        print('Building new task...')
        self.task = task.Task(self.dataset_config_set, self.template_config, self.model_config, self.criterion_config)
        self.optimizer = optimizers.load_optimizer(optimizer_config.optimizer_name)()(
            params = self.task.parameters(), optimizer_config = optimizer_config)

        if not resume.lower() == 'none': self.load_checkpoint(resume)

        self.all_params = self.task.parameters()
        total_params = sum([functools.reduce(lambda x, y: x * y, tsr.shape) for tsr in self.all_params if tsr.size() and tsr.requires_grad])
        print(f'Model total parameters: {total_params}.')
        # input('stop')
        if torch.cuda.is_available():
            self.task, self.optimizer = self.try_cuda(self.task, self.optimizer)

        self.md5_name = hashlib.md5(self.str_configs().encode()).hexdigest()

        if self.hp.log_file == True: self.hp.log_file = self.get_log_path()
                  
    def __call__(self):
        while self.cur_epoch < self.hp.max_epoch:
            try:
                if self.optimizer_config.lr_update:
                    utils.adjust_learning_rate(self.optimizer, self.cur_epoch, self.hp.max_epoch, self.optimizer_config.lr)
                else:
                    utils.set_learning_rate(self.optimizer, self.hp.lr, False)
                _ = self.train_epoch_dataset_first(self.cur_epoch, 'trn')
            except KeyboardInterrupt:
                self.handle_exception()
                print('Exit control menu.')
            # nni.report_intermediate_result(0.5)
            # if self.cur_epoch % self.config.val_interval == 0:
            try:
                ret_val, val_loss, val_eval = self.train_epoch_dataset_first(self.cur_epoch, 'val')
                ret_tst, tst_loss, tst_eval = self.train_epoch_dataset_first(self.cur_epoch, 'tst')
                if self.hp.evaluation == 'loss':
                    self.undec = self.recoder.push_loss(self.cur_epoch, self.undec, val_loss, ret_tst)
                    if self.hp.nni:
                        nni.report_intermediate_result(tst_loss)
                elif self.hp.evaluation == 'acc':
                    self.undec = self.recoder.push_eval(self.cur_epoch, self.undec, val_eval, ret_tst)
                    if self.hp.nni:
                        nni.report_intermediate_result(tst_eval)
                else:
                    raise ValueError('Unknown evaluation.')
                if self.undec == 0: self.save_checkpoint()
            except KeyboardInterrupt:
                print(f'Skipping val and test for ctrl + c detected.')

            self.cur_epoch += 1
            if self.undec >= self.hp.stop_val_dec: 
                print('Val_loss hasn\'t decrease in the last [{}] epoches, stop training early.'.format(self.hp.stop_val_dec))
                break
                
        if self.hp.evaluation == 'loss':
            fin_epoch, fin_loss = self.recoder.pop_via_loss()
            if self.hp.nni:
                nni.report_final_result(fin_loss)
            print(f'[{self.cur_epoch}] epoches complete, output results = {fin_loss} at epoch [{fin_epoch}], seed = {self.hp.seed}.')
        elif self.hp.evaluation == 'acc':
            fin_epoch, fin_eval = self.recoder.pop_via_eval()
            if self.hp.nni:
                nni.report_final_result(fin_eval)
            print(f'[{self.cur_epoch}] epoches complete, output results = {fin_eval} at epoch [{fin_epoch}], seed = {self.hp.seed}.')
        else:
            raise ValueError('Unknown evaluation.')
        self.evaluate()
        
    def train_epoch_dataset_first(self, epoch_idx, tvt):
        assert tvt in ['trn', 'val', 'tst'], 'tvt must be chosen from [trn, val, tst].'
        if tvt == 'trn': self.task.train()
        else: self.task.eval()

        total_losses = 0
        total_acc = 0
        ret_all = {}

        max_batch = max([len(v.dataset.loader[tvt]) for k, v in self.dataset_config_set.items()])

        for data_name, data_config in self.dataset_config_set.items():
            batch_cnt = max_batch if self.hp.same_num_batch else len(data_config.dataset.loader[tvt])
            ret_dataset, loss_weighted, acc_weighted = self.dataset_forward_backward(epoch_idx, data_name, data_config, batch_cnt, tvt)
            for k, v in ret_dataset.items(): ret_all[k] = v
            total_losses += loss_weighted * data_config.weight
            total_acc += acc_weighted * data_config.weight
        return ret_all, total_losses, total_acc

    def dataset_forward_backward(self, epoch_idx, data_name, data_config, max_batch, tvt):
        ret_all = {}
        loss_weighted = 0
        acc_weighted = 0
        idx = 0
        extra_input = None

        for idx, (midx, mwei) in enumerate(zip(data_config.model_idx, data_config.model_weight)):
            ret_model = self.model_forward_backward(epoch_idx, data_name, data_config, max_batch, str(midx), tvt)
            ret_all[f'{data_name}_model_{midx}_acc'] = ret_model['acc']
            ret_all[f'{data_name}_model_{midx}_loss'] = ret_model['loss_detach']
            acc_weighted = ret_model['acc'] * mwei
            loss_weighted += ret_model['loss_detach'] * mwei

        return ret_all, loss_weighted, acc_weighted

    def model_forward_backward(self, epoch_idx, data_name, data_config, max_batch, model_idx, tvt):
        # process a dataset - model idx pair
        batch_counter = 0
        ret_avg = {}
        backward_loss = []
        extra_input = None
        inner_recoder = utils.statistic()

        while batch_counter < max_batch:
            for batch_idx, cur_batch in enumerate(data_config.dataset.loader[tvt]):
                # warm up
                if self.cur_epoch == 0 and self.hp.warmup > 0 and batch_idx < self.hp.warmup:
                    utils.set_learning_rate(self.optimizer, self.hp.lr * batch_idx / self.hp.warmup, False)
                if tvt == 'trn':
                    bret = self.trn_batch(data_config, *cur_batch, extra_input, model_idx, batch_counter, backward_loss)
                else:
                    bret = self.eva_batch(data_config, *cur_batch, extra_input, model_idx)

                extra_input = utils.manage_extra_output(bret['extra_output'])

                inner_recoder.push(bret) # model statistic -> only work in this function
                # log results
                log_interval = self.hp.log_interval if tvt == 'trn' else max_batch
                if (batch_counter + 1) % log_interval == 0 or batch_counter == (max_batch - 1):
                    pp = inner_recoder.pop()
                    tmp_avg = self.print_batches(epoch_idx, data_name, pp, batch_counter + 1, max_batch, tvt, model_idx)
                    for k, v in tmp_avg.items():
                        if not k in ret_avg: ret_avg[k] = []
                        ret_avg[k].append(v)

                # for same batch option
                batch_counter += 1
                if not batch_counter < max_batch: break

        ret_avg['loss_detach'] = sum(ret_avg['loss_detach']) / len(ret_avg['loss_detach'])
        ret_avg['correct'] = sum(ret_avg['correct'])
        ret_avg['total'] = sum(ret_avg['total'])
        ret_avg['acc'] = ret_avg['correct'] / ret_avg['total']
        ret_avg['bpc'] = ret_avg['loss_detach'] / math.log(2)
        
        return ret_avg

    def trn_batch(self, data_config, src, wiz, tgt, extra_input, model_idx, batch_counter, backward_loss):
        self.task.train()
        if isinstance(self.cuda, str):
            cur_batch = [
                src.to(self.cuda) if src is not None else src,
                wiz.to(self.cuda) if wiz is not None else wiz,
                tgt.to(self.cuda) if tgt is not None else tgt,
            ]
        elif isinstance(self.cuda, int):
            cur_batch = [
                src.cuda() if src is not None and self.cuda > 0 else src,
                wiz.cuda() if wiz is not None and self.cuda > 0 else wiz,
                tgt.cuda() if tgt is not None and self.cuda > 0 else tgt,
            ]
        else:
            raise ValueError(f'Invalid value of self.cuda, got [{self.cuda}]')
        
        # forward
        task_ret = self.task(data_config, cur_batch, extra_input, model_idx, self.writer)
        backward_loss.append(task_ret['loss'])
        if batch_counter % self.hp.backward_interval == 0 or batch_counter == (max_batch - 1):
            loss = torch.sum(torch.cat(backward_loss))
            backward_loss.clear()

            if self.apex:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.hp.weight_decay > 0: self.task.apply(self.apply_weight_decay)
            if self.hp.clip > 0: torch.nn.utils.clip_grad_norm_(self.all_params, self.hp.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return task_ret

    def eva_batch(self, data_config, src, wiz, tgt, extra_input, model_idx):
        self.task.eval()
        if isinstance(self.cuda, str):
            cur_batch = [
                src.to(self.cuda) if src is not None else src,
                wiz.to(self.cuda) if wiz is not None else wiz,
                tgt.to(self.cuda) if tgt is not None else tgt,
            ]
        elif isinstance(self.cuda, int):
            cur_batch = [
                src.cuda() if src is not None and self.cuda > 0 else src,
                wiz.cuda() if wiz is not None and self.cuda > 0 else wiz,
                tgt.cuda() if tgt is not None and self.cuda > 0 else tgt,
            ]
        else:
            raise ValueError(f'Invalid value of self.cuda, got [{self.cuda}]')
        # forward
        with torch.no_grad():
            task_ret = self.task(data_config, cur_batch, extra_input, model_idx, self.writer)
            return task_ret

    def print_batches(self, epoch_idx, dataset_name, task_ret_set, batch_cnt, batch_num, tvt, model_idx):
        ret = {}
        ret['loss_detach'] = task_ret_set['loss_detach'].cpu().numpy()
        ret['correct'] = task_ret_set['correct'].cpu().numpy()
        ret['total'] = task_ret_set['total'].cpu().numpy()
        ret['acc'] = ret['correct'] / ret['total']
        ret['bpc'] = ret['loss_detach'] / math.log(2)
        
        time_now = time.time()
        time_span = time_now - self.time_start
        self.time_start = time_now
        tmp = '|Epoch {}|{}->{}|lr {:2.2e}|batch {}/{}|loss {:4.3f}|bpc {:4.3f}|acc {:4.2f}|correct {:5d}|total {:5d}|{:3.2f}s|'.format(
            str(epoch_idx).rjust(2), self.dataset_config_set[dataset_name].dataset.task_type, dataset_name.rjust(13), 
            self.optimizer.param_groups[0]['lr'], str(batch_cnt).rjust(4), str(batch_num).rjust(4), ret['loss_detach'], 
            ret['bpc'], ret['acc'] * 100, int(ret['correct']), int(ret['total']), time_span)
        
        if self.hp.log_print: print(tmp)
        if self.hp.log_file:
             with open(self.log_path, 'a') as f: f.write(tmp + '\n')

        return ret

    def save_checkpoint(self):
        if not self.hp.save_model:
            print(f'New checkpoint is ready but task can\'t be saved since hp.save_model = False')
            return
        file_name = self.get_pyl_path()
        if self.cuda > 1:
            model_dict = self.task.module.state_dict()
        else:
            model_dict = self.task.state_dict()
        optim_dict = self.optimizer.state_dict()
        torch.save({'cur_epoch': self.cur_epoch,
                    'state_dict': model_dict,
                    'recoder': self.recoder,
                    'optimizer': optim_dict,
                    'undec': self.undec,
                    'seed': self.seed,
                    }, 
                    file_name)
        print('Task has been saved at [{}]'.format(file_name))        

    def load_checkpoint(self, file_name):
        file_name = file_name.lower()
        if file_name == 'true':
            file_name = self.get_pyl_path()
        else:
            file_name = "saved_tasks/{}".format(file_name)
        if not os.path.exists(file_name):
            raise ValueError('Cannot find specific task [{}].'.format(file_name))

        print('Loading task from [{}]'.format(file_name))
        checkpoint = torch.load(file_name)
        self.task.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.cur_epoch = checkpoint['cur_epoch']
        self.recoder = checkpoint['recoder']
        self.undec = checkpoint['undec']
        self.seed = checkpoint['seed']
        if self.cuda > 0:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    def handle_exception(self):
        print('-' * 89)
        print('Exiting from epoch {} early, command:'.format(self.cur_epoch))
        getch = utils.Getch()
        command = '/'
        while not command == 'q' or command == '1':
            print('Choose a selection:')
            # print('0: Continue training -> next epoch directly, without val and test.')
            print('1: Continue training -> next epoch, with val and test of current epoch.')
            print('2: Print seed.')
            print('3: Set learning rate.')
            print('4: Save model.')
            print('5: Enable Learning_rate update.')
            print('6: Disable Learning_rate update.')
            print('7: Set cur Epoch.')
            print('8: Set max Epoch.')
            print('9: Exit training and print test results.')
            print('Q: Quit.')
            print('-' * 89)
            command = getch()
            if not isinstance(command, str): command = str(command)   
            print('Your input is:', command)
            if '0' in command:
                pass
            elif '1' in command:
                print('1:Epoch [{}] starts...'.format(self.cur_epoch))
                break
            elif '2' in command:
                print('2:print seed...')
                print(f'Seed = {self.seed}')

            elif '3' in command:
                tmp = input('3:Input new learning rate:')
                new_lr = float(tmp)
                utils.set_learning_rate(self.optimizer, new_lr)

            elif '4' in command:
                print('4:Saving model, input the save path (less 5 characters for default).')
                file_path = input()
                if len(file_path) < 5: file_path = None
                self.save_checkpoint()

            elif '5' in command:
                self.optimizer_config.lr_update = True
                print('5:Enable Learning_rate update.')
                print('Done')

            elif '6' in command:
                self.optimizer_config.lr_update = False
                print('5:Disable Learning_rate update.')
                print('Done')

            elif '7' in command:
                print('7:Set cur Epoch.')
                ce = input('Input cur epoch:')
                self.cur_epoch = int(ce)
                print('Done')

            elif '8' in command:
                print('8:Set cur Epoch.')
                ce = input('Input max epoch:')
                self.config.max_epoch = int(ce)
                print('Done')
            elif '9' in command:
                print('9:Exit training and print test results.')
                self.config.max_epoch = self.cur_epoch
                print('Done')
            elif 'q' in command or 'Q' in command:
                print('Q:Exit.')
                sys.exit(1)
            else:
                print('Unknown command.')
        print('Wait up to 5 seconds...')
        time.sleep(5)

    def str_configs(self):
        ret = ''
        ret += json.dumps(self.hp.__dict__) + '#######'

        for k, v in self.template_config.items():
            if 'is_none' == k:
                ret += f'{k}: {v},'
            else:
                ret += json.dumps(v.__dict__)
            ret += '#######'

        for k, v in self.dataset_config_set.items():
            for kk, vv in v.__dict__.items():
                if not kk == 'dataset':
                    ret += f'{kk}: {vv},'
            ret += '#######'

        for k, v in self.model_config.items():
            ret += json.dumps(v.__dict__)
            ret += '#######'

        for k, v in self.criterion_config.items():
            ret += json.dumps(v.__dict__)
            ret += '#######'

        ret += json.dumps(self.optimizer_config.__dict__) + '#######'

        return ret   

    def get_pyl_path(self):
        if not os.path.exists('saved_tasks'): 
            os.mkdir('saved_tasks')
        pre = f'saved_tasks/task_{self.hp.config_name}_{self.md5_name}.pyl'
        return pre

    def get_log_path(self):
        if not os.path.exists('saved_logs'): os.mkdir('saved_logs')
        pre = f'saved_logs/log_{self.hp.config_file}_{self.md5_name}.txt'
        return pre

    def try_apex(self, power):
        if not power: 
            print('Double precision mode [float32]')
            return False
        ret = False
        try:
            import apex
            from apex import amp
            ret = True
        except:
            ret = False
        if ret: print('Convert to hybrid precision mode [float16 - float32]')
        else: print('Do not support single precision mode [float16]')
        return ret

    def try_cuda(self, model, optimizer):
        if isinstance(self.cuda, int):
            if self.cuda < 1: return model, optimizer
            else: model = model.cuda()
        elif isinstance(self.cuda, str):
            model = model.to(self.cuda)
        else:
            raise ValueError(f'Fail to use cuda for self.cuda = [{self.cuda}]')

        if self.apex: 
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level = 'O1')
        if isinstance(self.cuda, int) and self.cuda > 1: model = nn.DataParallel(model)
        return model, optimizer

    def apply_weight_decay(self, m):
        if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)) and m.weight.grad is not None:
            m.weight.grad += m.weight * self.hp.weight_decay

    def evaluate(self, dataset_config_set = None):
        path = 'saved_logs/'
        if not os.path.exists(path): os.mkdir(path)
        if dataset_config_set is None:
            dataset_config_set = self.dataset_config
        
        for batch_name, batch_data in dataset_config_set.items():
            for mi in batch_data.model_idx:
                tmp = time.localtime(time.time())
                file_name = os.path.join(path, f'{batch_name}_{tmp[1]}_{tmp[2]}_{tmp[3]}_{tmp[4]}.txt')
                logits = []
                targets = []
                for (src, wiz, tgt) in batch_data.dataset.loader['tst']:
                    cur_batch = [
                        src.cuda() if src is not None and self.cuda > 0 else src,
                        wiz.cuda() if src is not None and self.cuda > 0 else wiz,
                        tgt.cuda() if src is not None and self.cuda > 0 else tgt,
                    ]
                    tmp = self.task(task_type = batch_data.dataset.task_type, 
                                    dataset_name = batch_name, 
                                    batch = cur_batch, 
                                    extra_input = None,
                                    writer = None, 
                                    tvt = 'tst',
                                    model_idx = mi,
                                   )
                    logits.append(tmp['logits'].cpu().numpy())
                    targets.append(tgt.cpu().numpy())
                logits = np.concatenate(logits, 0)
                targets = np.expand_dims(np.concatenate(targets), axis = 1)
                forsave = np.concatenate([targets, logits], axis = 1)
                np.savetxt(file_name, forsave)
                utils.plot_roc(file_name)





















