from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate,StandardScaler
from utils.metrics import metric
from utils.tre_loss2 import  tre_loss2
from tslearn.metrics import dtw, dtw_path
from utils.dilate_loss import dilate_loss

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,SubsetRandomSampler

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        #每种数据要用那种读取方式
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            # 'ETTh2':Dataset_ETT_hour,
            'ETTh2':Dataset_Custom,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,#自定义
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'JN':Dataset_Custom,
            'traffic':Dataset_Custom,
            'SY':Dataset_Custom
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            #shuffle训练时洗牌
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        start_index = 0
        interval = 96
        num_indices = 500
        indices = [start_index + interval * i for i in range(num_indices)]
        sample = SubsetRandomSampler(indices)
        data_loader = DataLoader(  #Dataloader函数 数据分组 把训练数据分成多个小组 每次抛出一组数据，直到把所有数据抛出
            data_set,
            batch_size=batch_size,
            # shuffle=shuffle_flag, #是否打乱
            num_workers=args.num_workers, #进程数
            drop_last=drop_last,
            sampler=sample)#丢弃不足一个batch_size的样本
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        self.scaler = StandardScaler()
        total_mseloss = []
        total_maeloss = []
        total_smapeloss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):

            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # pred = pred * 107.89299774  # 反归一化
            # true = true * 107.89299774
            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            pred_np = pred.detach().cpu().numpy()
            true_np = true.detach().cpu().numpy()
            pred_np = np.array(pred_np)
            true_np = np.array(true_np)
            pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
            true_np = true_np.reshape(-1, true_np.shape[-2], true_np.shape[-1])
            mae, mse, rmse, mape, mspe, smape,dtw,tdi = metric(pred_np, true_np)

            total_mseloss.append(mse)
            total_maeloss.append(mae)
            total_smapeloss.append(smape)

        total_mseloss = np.average(total_mseloss)
        total_maeloss = np.average(total_maeloss)
        total_smapeloss = np.average(total_smapeloss)
        self.model.train()
        return total_mseloss,total_maeloss,total_smapeloss


    def train(self, setting,alpha=1,gamma=0.001):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)#训练步数 根据batch指定 # 24453/32
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)#提前停止策略
        
        model_optim = self._select_optimizer()#优化器
        #损失函数
        criterion =  self._select_criterion()#损失函数
        if self.args.use_amp:#自动缩放梯度
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            mae_loss = []
            smape_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):#DatasetCustom中getitem函数
                iter_count += 1                                                           #循环32次
                # 32 96 12    32 72 12    32 96 4    32 72 4
                model_optim.zero_grad()#梯度清零
                pred, true = self._process_one_batch(#得到预测值和真实值
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # pred = pred * 107.89299774
                # true = true * 107.89299774

                # loss = criterion(pred, true)
                if (self.args.loss_type == 'mse'):
                    loss_mse = criterion(pred, true)
                    loss = loss_mse
                if (self.args.loss_type == 'tre'):
                    loss_tre = tre_loss2(true, pred, alpha, device)
                    loss = loss_tre

                if (self.args.loss_type == 'dilate'):
                    loss, loss_shape, loss_temporal = dilate_loss(true, pred, alpha, gamma, device)

                pred_np = pred.detach().cpu().numpy()
                true_np = true.detach().cpu().numpy()
                pred_np = np.array(pred_np)
                true_np = np.array(true_np)
                pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
                true_np = true_np.reshape(-1, true_np.shape[-2], true_np.shape[-1])
                mae, mse, rmse, mape, mspe, smape, dtw, tdi = metric(pred_np,true_np)

                train_loss.append(mse.item())

                mae_loss.append(mae.item())
                smape_loss.append(smape.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | mse_loss: {2:.7f} | mae_loss: {3:.7f} | smape_loss:{4:.7f}".format(i + 1,
                                                                    epoch + 1, mse.item(), mae.item(), smape.item()))
                    print("loss: ",loss.item())
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            mae_loss = np.average(mae_loss)
            smape_loss = np.average(smape_loss)
            print(
                "Epoch: {0}, Steps: {1} | Train mse_Loss: {2:.7f} | Train mae_loss: {3:.7f} | Train smape_loss: {4:.7f} "
                .format(
                    epoch + 1, train_steps, train_loss, mae_loss, smape_loss))
            # if epoch != 0 and epoch%1 == 0:
            vali_mseloss, vali_maeloss, vali_sampeloss = self.vali(vali_data, vali_loader, criterion)
            test_mseloss, test_maeloss, test_smapeloss = self.vali(test_data, test_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1}"
                "\n Vali mse_Loss: {2:.7f} | vali mae_loss: {3:.7f} | vali smape_loss: {4:.7f}"
                "\n Test mse_Loss: {5:.7f} | test mae_loss: {6:.7f} | test sampe_loss: {7:.7f}"
                .format(
                    epoch + 1, train_steps, vali_mseloss, vali_maeloss,
                    vali_sampeloss, test_mseloss, test_maeloss, test_smapeloss))
            early_stopping(vali_mseloss, self.model, path,self.args.loss_type)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+self.args.loss_type+'_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))#d读取最佳参数
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + self.args.loss_type+'_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))  # d读取最佳参数
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            # print(i)
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # pred = pred * 107.89299774
            # true = true * 107.89299774
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, sampe,dtw,tdi = metric(preds, trues)
        print('mse:{}, mae:{}, smape:{},dtw:{},tdi:{}'.format(mse, mae,sampe,dtw,tdi))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+self.args.loss_type+'_checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:#以0为初始化进行构建 32 24 12
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float() #
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        #最后24个是0 48个已知加24个初始化的
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #batch_x为特征数据 32 96 12 batch_x_mark为时间特征 32 96 4
                                                                                   #outputs 32 24 12 预测值
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y #预测值与真实值
