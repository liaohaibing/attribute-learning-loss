import numpy as np

# 加载文件
fname="F:/tre_loss/code/data/PEM/traffic.txt"
rawdata=np.loadtxt(fname, delimiter=',')
print('rawdata==', rawdata.shape)
tf_data=rawdata[:,0]/100
X_min = np.min(tf_data)
X_max = np.max(tf_data)
## x_max= 0.033
## x_min=0.809
print("min=",X_min)
print("max=",X_max)
length = len(tf_data)
nb=round(length*0.8)
tf_train=tf_data[0:nb]
tf_test=tf_data[nb:]
X=[]
for i in range(len(tf_train) - 144 + 1):
    temp = tf_train[i:i + 144]
    X.append(temp)
N=round(len(X)*0.8)
X = np.stack(X)
train_input = X[0:N, 0:72]
train_target = X[0:N, 72:144]
val_input = X[N:len(X), 0:72]
val_target = X[N:len(X), 72:144]
train_name='PEM72'+"_train_input.txt"
train_name2 = 'PEM72' + "_train_target.txt"
val_name = 'PEM72' + "_val_input.txt"
val_name2 = 'PEM72' + "_val_target.txt"
np.savetxt(train_name, train_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(train_name2, train_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(val_name, val_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(val_name2, val_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔

Y=[]
for i in range(len(tf_test) - 144 + 1):
    temp = tf_test[i:i + 144]
    Y.append(temp)
Y = np.stack(Y)
test_input = Y[:, 0:72]
test_target = Y[:, 72:144]
test_name='PEM72'+"_test_input.txt"
test_name2 ='PEM72' + "_test_target.txt"
np.savetxt(test_name, test_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(test_name2, test_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔


