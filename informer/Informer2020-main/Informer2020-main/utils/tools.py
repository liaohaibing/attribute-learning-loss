import numpy as np
import torch
from exp.exp_basic import Exp_Basic
#loss_type="dilate"  #################################################
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}

    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 0.0005, 15: 1e-7, 20: 5e-8

        }
    if epoch in lr_adjust.keys():
        # lr = lr_adjust[epoch]
        lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, loss_type):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,loss_type)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,loss_type)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, loss_type):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+loss_type+'_checkpoint.pth')##########################################
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0. #指定初始化均值和标准差
        self.std = 1.
        self.min = 0.
        self.max_min = 0.
    
    def fit(self, data):
        # self.mean = data.mean(0) #0代表压缩行，对列求均值
        # self.std = data.std(0)
        self.min = data.min(0)
        self.max_min = data.max(0) - self.min
        return  self.min,self.max_min
    #转化为tensor格式
    def transform(self, data):
        # mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        # std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_min = torch.from_numpy(self.max_min).type_as(data).to(data.device) if torch.is_tensor(data) else self.max_min
        # return (data - mean) / std#返回预处理结果
        return (data - min)/max_min
    def inverse_transform(self, data):
        # mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        # std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_min = torch.from_numpy(self.max_min).type_as(data).to(data.device) if torch.is_tensor(data) else self.max_min
        # if data.shape[-1] != min.shape[-1]:
        #     min = min[-1:]
        #     max_min= max_min[-1:]
        #  return (data * std) + mean
        return  data * (max_min) + min
