import torch
from statsmodels.tsa.seasonal import STL
import numpy as np

###-------Loss function design based on time series attribute learning （Attri_Loss)------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def std_loss(targets,outputs,alpha1,alpha2,alpha3,device):
	#------- value loss  ------------
	sq_error = w_mse(targets,outputs,device)
	value_loss=torch.mean(sq_error)
    #------------STL-------------------
	batch_size, len = targets.shape[0:2]
	T1 = np.ones((batch_size,len))
	T2 = np.ones((batch_size,len))
	S1 = np.ones((batch_size,len))
	S2 = np.ones((batch_size, len))
	x1=targets.cpu().detach().numpy().squeeze()
	x2=outputs.cpu().detach().numpy().squeeze()
	for i in range(batch_size):
		data1=x1[i,:]
		data2=x2[i,:]
		stl1 = STL(data1, period=12, trend=21, seasonal=13, low_pass=13, robust=True)  #
		stl2 = STL(data2, period=12, trend=21, seasonal=13, low_pass=13, robust=True)  #

		res1 = stl1.fit()
		res2 = stl2.fit()
		T1[i,:]=res1.trend
		T2[i, :] = res2.trend
		S1[i,:]=res1.seasonal
		S2[i, :] = res2.seasonal
	#--------trend and seasonal loss ----------------

	#trend_error=np.mean(np.abs(T1-T2))#MAE
	
	corr = np.corrcoef(T1, T2)
	d_corr = corr[0:batch_size, batch_size::]
	trend_error = 1 - np.diag(d_corr, 0)
	trend_loss = torch.mean(torch.tensor(trend_error, dtype=torch.float32).to(device))

	#seasonal_error =np.mean(np.abs(S1 - S2))#MAE
	
	num = np.dot(S1, np.array(S2).T)  # 向量点乘
	denom = np.linalg.norm(S1, axis=1).reshape(-1, 1) * np.linalg.norm(S2, axis=1)  # 求模长的乘积
	res = num / denom
	s=np.diag(res,0)
	temp=s.copy()
	temp[np.isneginf(temp)] = 0
	S=0.5 + 0.5 * temp
	seasonal_error=1-S
	seasonal_loss=  torch.mean(torch.tensor(seasonal_error, dtype=torch.float32).to(device))

	loss = trend_loss*alpha1+seasonal_loss*alpha2+value_loss*alpha3
	return loss
def w_mse(targets,outputs,device):
	batch_size, len = targets.shape[0:2]
	target1= targets[:,1:len,:].to(device)
	target2= targets[:,0:len-1,:].to(device)
	output1 = outputs[:, 1:len, :].to(device)
	output2 = outputs[:, 0:len - 1, :].to(device)
	sigma_matrix=(target1-target2)*(output1-output2).to(device)
	sigma=torch.sign(sigma_matrix).to(device)
	newtargets=targets[:,1:len,:].to(device)
	newoutputs=outputs[:,1:len,:].to(device)
	W=(1.01+torch.abs(newoutputs-newtargets)/(torch.abs(newoutputs+newtargets)+0.00001))**(1-sigma).to(device)
	w_error=W*torch.abs(newoutputs-newtargets)
	return w_error
def derivatives(input,device):
	batch_size,len =input.shape[0:2]
	input2 = input[:,2:len,:].to(device)
	input1= input[:,0:len-2,:].to(device)
	D= input2-input1
	return D


