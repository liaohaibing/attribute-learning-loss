import os
import numpy as np
import argparse
import  torch
from exp.exp_informer import Exp_Informer
from utils.metrics import metric
from models.model import Informer, InformerStack
parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
parser.add_argument('--model', type=str,default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
# parser.add_argument('--data', type=str, default='ETTh2', help='data')
# parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
# parser.add_argument('--data', type=str, default='JN', help='data')#####选数据集地方
# parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='JN.csv', help='data file')#####选数据集地方
# parser.add_argument('--data', type=str, default='ECL', help='data')#####选数据集地方
# parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='ECL.csv', help='data file')#####选数据集地方
# parser.add_argument('--data', type=str, default='traffic', help='data')#####选数据集地方
# parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')#####选数据集地方
parser.add_argument('--data', type=str, default='SY', help='data')#####选数据集地方
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='SY.csv', help='data file')#####选数据集地方

parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')#训练轮数
parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')

parser.add_argument('--loss_type', type=str, default='tre',help='loss function')
args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

#多卡操作
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
#M要预测多少时间点
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'HUFL','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},#M指预测多少时间
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'JN': {'data': 'JN.csv', 'T': 'F_PM25', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'traffic': {'data': 'traffic.csv', 'T': 'traffic', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'SY':{'data': 'SY.csv', 'T': 'SY', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]}#min=0；max=1.812118
    # 'ETTh2':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}#  增加数据参数
#取出参数
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]#enc_in 每组数据12维度

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]#循环3次
args.detail_freq = args.freq #h 代表以小时为间隔
args.freq = args.freq[-1:]
exp = Exp_Informer(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
    args.data,args.features,args.seq_len,args.label_len, args.pred_len,args.d_model,args.n_heads, args.e_layers,args.d_layers,
    args.d_ff, args.attn,args.factor, args.embed,args.distil,args.mix, args.des,ii)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    test_data, test_loader = exp._get_data(flag='test')

    model_dict = {
        'informer': Informer,
        'informerstack': InformerStack,
    }
    if args.model == 'informer' or args.model == 'informerstack':
        e_layers = args.e_layers if args.model == 'informer' else args.s_layers
        model = model_dict[args.model](args.enc_in,
                args.dec_in,
                args.c_out,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.factor,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.dropout,
                args.attn,
                args.embed,
                args.freq,
                args.activation,
                args.output_attention,
                args.distil,
                args.mix,
                device
            ).float().to(device)
        path = os.path.join(args.checkpoints, setting)
        best_model_path = path + '/' + args.loss_type + '_checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))  # d读取最佳参数
        model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        # print(i)
        # pred, true = exp._process_one_batch(
        #     test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        if args.padding == 0:  # 以0为初始化进行构建 32 24 12
            dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()  #
        elif args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
        # 最后24个是0 48个已知加24个初始化的
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        # encoder - decoder
        if args.use_amp:
            with torch.cuda.amp.autocast():
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp,batch_y_mark)  # batch_x为特征数据 32 96 12 batch_x_mark为时间特征 32 96 4
                # outputs 32 24 12 预测值
        # if args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
        # pred = outputs * 107.89299774
        # true = batch_y * 107.89299774
        preds.append(outputs.detach().cpu().numpy())
        trues.append(batch_y.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    jn_max=398
    jn_min=2
    ett_max=107.89299774
    ett_min=0
    ecl_max=6035
    ecl_min=0


    # preds = preds * 0.01
    # trues = trues * 0.01
    mae, mse, rmse, mape, mspe, sampe,dtw,tdi = metric(preds, trues)
    print('mse:{}, mae:{}, smape:{},dtw:{},tdi:{}'.format(mse, mae, sampe,dtw,tdi))

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)