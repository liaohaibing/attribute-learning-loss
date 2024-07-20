import torch

def tre_loss(targets,outputs,alpha,device):
	sq_error = (targets - outputs) ** 2
	error1=torch.mean(sq_error)
	d_error = (derivatives(targets,device) - derivatives(outputs,device))**2
	error2=torch.mean(d_error)
	add_error = error1+alpha*error2
	loss= add_error
	return loss
def derivatives(input,device):
	batch_size, len = input.shape[0:2]
	input2 = input[:, 2:len, :].to(device)
	input1 = input[:, 0:len - 2, :].to(device)
	D = input2 - input1
	# D=torch.zeros((batch_size,len,1)).to(device)
	# for k in range(batch_size):
	# 	temp=input[k,:,:].view(-1,1)
	# 	Dk = torch.zeros((len,1)).to(device)
	# 	for i in range(2,len-2):
	# 		#Dk[i,0]=((temp[i,0]-temp[i-1,0])+0.5*(temp[i+1,0]-temp[i-1,0]))/2
	# 		#Dk[i, 0] = 2*temp[i, 0] - temp[i - 1, 0]-temp[i + 1, 0]
	# 		Dk[i, 0] = (temp[i, 0] - temp[i - 1, 0])+ (temp[i, 0]-temp[i - 2, 0]) +(temp[i +1, 0]-temp[i, 0])+ (temp[i + 2, 0]-temp[i, 0])
	# 	Dk[0,0]=Dk[2,0]
	# 	Dk[1, 0] = Dk[2, 0]
	# 	Dk[len-1,0]=Dk[len-3,0]
	# 	Dk[len-2,0] = Dk[len-3, 0]
	# 	D[k:k+1,:,:]=Dk
	return D