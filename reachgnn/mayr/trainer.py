import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import copy
import torch
from torch_scatter import scatter
from torch.optim.lr_scheduler import MultiStepLR


class Trainer_Mayr(object):
    def __init__(self, PATH=None, model=None, train_loader=None, val_loader=None, 
                 val_loader1=None, val_loader2=None, test_loader=None, test_loader1=None,
                 test_loader2=None, criterion=None, optimizer=None, device=None, lr_scheduler=None):
        if device:
            model.to(device)

        self.PATH = PATH
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader1 = val_loader1
        self.val_loader2 = val_loader2
        self.test_loader = test_loader
        self.test_loader1 = test_loader1
        self.test_loader2 = test_loader2
        self.criterion = criterion
        self.optimizer = optimizer
         #milestones = list(range(200, 400, 20)) + \
        #    list(range(400, 600, 40)) + list(range(600, 1000, 80))
        #self.lr_scheduler = MultiStepLR(
        #    self.optimizer, milestones=milestones, gamma=0.8, verbose=False) if not lr_scheduler else lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                mode='min', factor=0.5, patience=10, threshold=0.001, threshold_mode='rel', 
                        cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        y_true = train_loader.dataset.data['rxn'].y
        self.y_mean, self.y_std = y_true.mean(), y_true.std()

        self.iterations = 0
        self.epoch_ = 0
        self.device = device
        self.history = dict()
        self.min_train_loss = 20
        self.min_val_loss = 20

    def train(self, data_iter, d_step=50, print_log=False):
        "Standard Training and Logging Function"
        import time
        start = time.time()
        total_loss = 0
        tmp_loss = 0
        num_samples = 0
        tmp_num = 0
        time_cost = 0
        for i, batch in enumerate(data_iter):
            if self.device:
                batch.to(self.device)
            _num = batch['rxn'].y.size(0)
            y_true = batch['rxn'].y.unsqueeze(0).T
            y_true = (y_true - self.y_mean) / self.y_std

            self.optimizer.zero_grad()
            pred, logvar = self.model(batch)
            loss = self.criterion(pred, y_true.float())
            loss = (1 - 0.1) * loss.mean() + 0.1 * \
                (loss * torch.exp(-logvar) + logvar).mean()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters, max_norm=1, norm_type=2)
            self.optimizer.step()

            tmp_loss += loss.detach().item()*_num
            tmp_num += _num
            if i > 0 and i % d_step == 0 or _num != data_iter.batch_size or tmp_num == len(data_iter.dataset):
                elapsed = time.time() - start
                if print_log:
                    print("    **Train Epoch Step: %d Loss: %f Cost Time: %.4fs" %
                          (i, tmp_loss / tmp_num, elapsed))
                start = time.time()
                total_loss += tmp_loss
                num_samples += tmp_num
                time_cost += elapsed
                tmp_loss, tmp_num = 0, 0

        self.iterations += i
        self.lr_scheduler.step(total_loss / num_samples)
        return total_loss / num_samples, time_cost

    def evalution(self, data_iter, d_step=50, print_log=False):
        "Standard Training and Logging Function"
        import time
        start = time.time()
        total_loss = 0
        tmp_loss = 0
        num_samples = 0
        tmp_num = 0
        time_cost = 0
        for i, batch in enumerate(data_iter):
            if self.device:
                batch.to(self.device)
            _num = batch['rxn'].y.size(0)
            y_true = batch['rxn'].y.unsqueeze(0).T
            y_true = (y_true - self.y_mean) / self.y_std
            with torch.no_grad():
                pred, logvar = self.model(batch)
            loss = self.criterion(pred, y_true.float())
            loss = (1 - 0.1) * loss.mean() + 0.1 * \
                (loss * torch.exp(-logvar) + logvar).mean()

            tmp_loss += loss.detach().item()*_num
            tmp_num += _num
            if i > 0 and i % d_step == 0 or _num != data_iter.batch_size or tmp_num == len(data_iter.dataset):
                elapsed = time.time() - start
                if print_log:
                    print("    **Val Epoch Step: %d Loss: %f Cost Time: %.4fs" %
                          (i, tmp_loss / tmp_num, elapsed))
                start = time.time()
                total_loss += tmp_loss
                num_samples += tmp_num
                time_cost += elapsed
                tmp_loss, tmp_num = 0, 0
        #self.iterations += i
        return total_loss / num_samples, time_cost

    def predict(self, data_iter, d_step=50, n_forward_pass=30, print_log=False):
        "Standard Training and Logging Function"
        import time
        start = time.time()
        total_loss = 0
        tmp_loss = 0
        num_samples = 0
        tmp_num = 0

        y_preds = torch.zeros(0, 1)
        y_trues = torch.zeros(0, 1)
        y_idxs = torch.zeros(0, 1)
        time_cost = 0
        self.model.eval()
        for i, batch in enumerate(data_iter):
            if self.device:
                batch.to(self.device)
            _num = batch['rxn'].y.size(0)
            y_true = batch['rxn'].y.unsqueeze(0).T
            y_idx = batch['rxn'].idx.unsqueeze(0).T
            y_trues = torch.cat((y_trues, y_true.cpu()))
            y_idxs = torch.cat((y_idxs, y_idx.cpu()))
            y_true = (y_true - self.y_mean) / self.y_std

            mean_lib = torch.zeros(_num, 0, device=device)
            var_lib = torch.zeros(_num, 0, device=device)
            self.MC_dropout(self.model)
            with torch.no_grad():
                for _ in range(n_forward_pass):
                    pred, logvar = self.model(batch)
                    mean_lib = torch.cat((mean_lib, pred), dim=1)
                    var_lib = torch.cat((var_lib, logvar), dim=1)
            mean_ = mean_lib.mean(dim=1).unsqueeze(0).T
            logvar_ = var_lib.mean(dim=1).unsqueeze(0).T
            loss = self.criterion(mean_, y_true.float())
            loss = (1 - 0.1) * loss.mean() + 0.1 * \
                (loss * torch.exp(-logvar_) + logvar_).mean()

            pred_ = mean_ * self.y_std + self.y_mean
            y_preds = torch.cat((y_preds, pred_.cpu()))

            tmp_loss += loss.detach().item()*_num
            tmp_num += _num
            if i > 0 and i % d_step == 0 or _num != data_iter.batch_size or tmp_num == len(data_iter.dataset):
                elapsed = time.time() - start
                if print_log:
                    print("    **Pred Epoch Step: %d Loss: %f Cost Time: %.4fs" %
                          (i, tmp_loss / tmp_num, elapsed))
                start = time.time()
                total_loss += tmp_loss
                num_samples += tmp_num
                time_cost += elapsed
                tmp_loss, tmp_num = 0, 0
        #self.iterations += i
        return y_preds, y_trues, y_idxs, total_loss / num_samples, time_cost

    @staticmethod
    def MC_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        pass

    def run(self, epochs=1, saved_step=200, log_d_step=50, print_step=5, print_train_log=False):
        self.saved_step = saved_step
        pre_epoch = self.epoch_
        for epoch in range(1, epochs + 1):
            self.epoch_ += 1
            self.val_loss = self.min_val_loss
            self.model.train()
            self.train_loss, train_time = self.train(
                self.train_loader, log_d_step, print_train_log)

            if self.epoch_ % print_step == 0:
                self.model.eval()
                self.val_loss, val_time = self.evalution(
                    self.val_loader, log_d_step, print_train_log)
                self.val_loss1, val_time1 = self.evalution(
                    self.val_loader1, log_d_step, print_train_log)
                self.val_loss2, val_time2 = self.evalution(
                    self.val_loader2, log_d_step, print_train_log)
                # self.test_loss, test_time = self.evalution(
                #    self.test_loader, log_d_step, print_train_log)

                # print("Eopch: %4d  10^3*Loss: %.6f, 10^3*Val_Loss: %.6f, train_time: %.4fs, val_time: %.4fs," %
                #  (self.epoch_, 1000*self.train_loss, 1000*self.val_loss, train_time, val_time))
                print("Eopch: %4d  Loss: %.6f, Val_Loss: (%.6f, %.6f, %.6f), \
                      train_time: %.4fs, val_time: (%.4fs, %.4fs, %.4fs)," %
                      (self.epoch_, self.train_loss, self.val_loss, self.val_loss1, self.val_loss2, 
                       train_time, val_time, val_time1, val_time2))
                #print("            Test_Loss: %.6f, Test_time: %.4fs," %(self.test_loss, test_time))
                print(
                    "---------------------------------------------------------------------------------")
                torch.cuda.empty_cache()
            self.history[self.epoch_] = {
                'train_loss': self.train_loss, 'val_loss': self.val_loss}

            if self.train_loss <= self.min_train_loss:
                self.min_train_loss = self.train_loss
                self.PATH_train_best = r'%s_train_best.pt' % (self.PATH[:-3])
                self.state_train_best = self.save_current_model(
                    self.PATH_train_best)

            if self.val_loss <= self.min_val_loss:
                self.min_val_loss = self.val_loss
                self.PATH_val_best = r'%s_val_best.pt' % (self.PATH[:-3])
                self.state_val_best = self.save_current_model(
                    self.PATH_val_best)

            if not (self.epoch_ % self.saved_step) or self.epoch_ - pre_epoch == epochs:
                self.PATH_last_saved = r'%s_%s.pt' % (
                    self.PATH[:-3], self.epoch_)
                self.state = self.save_current_model(self.PATH_last_saved)
                self.state = self.save_current_model(self.PATH)

            # for p in self.optimizer.param_groups:
            #    p['lr'] *= 0.985

    def save_current_model(self, save_path):
        state = {'epoch': self.epoch_, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'history': self.history, }
        torch.save(state, save_path)
        return state

    def load_state(self, state):
        self.epoch_ = state['epoch']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.history = state['history']

    def load_model(self, path=None):
        path = self.PATH if not path else path
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            self.load_state(checkpoint)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def save_Trainer(self, path=None):
        self.trainer_root = r'%s_trainer.pt' % (
            self.PATH[:-3]) if not path else path
        torch.save(self, self.trainer_root)
        print("Saved at", self.trainer_root)

    def load_Trainer(self, path=None):
        self.trainer_root = r'%s_trainer.pt' % (
            self.PATH[:-3]) if not path else path
        trainer = torch.load(self.trainer_root)
        self.copy_from_Trainer(trainer)
        print("Loaded from", self.trainer_root)

    def copy_from_Trainer(self, trainer):
        for k, v in trainer.__dict__.items():
            self.__dict__[k] = v


def LearningCurve(epoch, train_loss, val_loss, title='Learning Curve', figure_file=None):
    sns.set(style='whitegrid', palette='muted', color_codes=True)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    plt.plot(epoch, train_loss,  color='blue', label="Training Loss")
    plt.plot(epoch, val_loss,  color='green', label="Validation Loss")
    plt.tick_params(axis='both', which='major', labelsize=18)
    if title:
        plt.title(label=title, fontsize=28)
    plt.xlabel(xlabel=r'Epoch', fontsize=24)
    plt.ylabel(ylabel=r'Loss', fontsize=24)

    # plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距
    plt.legend(fontsize=18, loc='best')
    if figure_file != None:
        plt.savefig(figure_file, dpi=300)
    plt.show()


def Plot_True_vs_Pred(y_true, y_pred, title='Reaction rate constant: Pred vs. True', low=None, up=None, figure_file=None):
    sns.set(style='whitegrid', palette='muted', color_codes=True)
    low = low if low != None else min(y_true.min(), y_pred.min())-0.2
    up = up if up != None else max(y_true.max(), y_pred.max())+0.2
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    plt.scatter(y_true, y_pred,  color='green')
    plt.plot((low, up), (low, up), color='blue', linewidth=3)
    plt.tick_params(axis='both', which='major', labelsize=18)
    if title:
        plt.title(label=title, fontsize=28)
    plt.xlabel(xlabel=r'Experiment', fontsize=24)
    plt.ylabel(ylabel=r'Prediction', fontsize=24)
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    plt.text((low+up)/2+1, low+1, s='R$^2$ = %.4f\nMAE = %.4f \nRMSE = %.4f\n' % (R2, MAE, RMSE),
             fontdict={'size': '20', 'color': 'black'})

    # plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距
    if figure_file != None:
        plt.savefig(figure_file, dpi=300)
    plt.show()

