import numpy as np
import pandas as pd
from os.path import join as pjoin
#####训练数据读取###################################
train_file = pjoin('./ECL_data', 'ECL.csv')
usecols = ['date','MT_320']
df_data = pd.read_csv(train_file, usecols=usecols, error_bad_lines=False, low_memory=False,
                       encoding='utf-8', keep_default_na=False)
#df_data = df_data.sort_values(by='date')
ecl= pd.to_numeric(df_data['MT_320'], errors='coerce')
ecl_v=ecl.values
length = len(ecl_v)
# #------zero-mean normalization---
# x_mean = np.mean(ecl_v).astype(float)
# vari = np.sqrt((np.sum((ecl_v-x_mean)**2))/length)
# ecl_norm = (ecl_v-x_mean)/vari

#Linear normalization--Max-Min
X_min = np.min(ecl_v)
X_max = np.max(ecl_v)
## x_max= 6035
## x_min=0
print("min=",X_min)
print("max=",X_max)
ecl_norm = (ecl_v - X_min) / (X_max - X_min)
nb=round(length*0.82)
ecl_train=ecl_norm[0:nb]
ecl_test=ecl_norm[nb:]

# nb=round(length*0.82)
# ecl_train=ecl_v[0:nb]
# ecl_test=ecl_v[nb:]

X=[]
for i in range(len(ecl_train) - 144 + 1):
    temp = ecl_train[i:i + 144]
    X.append(temp)
N=round(len(X)*0.82)
X = np.stack(X)
train_input = X[0:N, 0:72]
train_target = X[0:N, 72:144]
val_input = X[N:len(X), 0:72]
val_target = X[N:len(X), 72:144]
train_name='ECL72'+"_train_input.txt"
train_name2 = 'ECL72' + "_train_target.txt"
val_name = 'ECL72' + "_val_input.txt"
val_name2 = 'ECL72' + "_val_target.txt"
np.savetxt(train_name, train_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(train_name2, train_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(val_name, val_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(val_name2, val_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔

Y=[]
for i in range(len(ecl_test) - 144 + 1):
    temp = ecl_test[i:i + 144]
    Y.append(temp)
Y = np.stack(Y)
test_input = Y[:, 0:72]
test_target = Y[:, 72:144]
test_name='ECL72'+"_test_input.txt"
test_name2 ='ECL72' + "_test_target.txt"
np.savetxt(test_name, test_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(test_name2, test_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔

print("finishing!")



# #####测试数据读取###################################
# test_file = pjoin('./ECL_data', 'ECL_test.csv')
# usecols = ['date','MT_320']
# df_data = pd.read_csv(test_file, usecols=usecols, error_bad_lines=False, low_memory=False,
#                        encoding='utf-8', keep_default_na=False)
# df_data = df_data.sort_values(by='date')
# ecl= pd.to_numeric(df_data['MT_320'], errors='coerce')
# ecl_v=ecl.values
# X=[]
# for i in range(len(ecl_v) - 96 + 1):
#     temp = ecl_v[i:i + 96]
#     X.append(temp)
# X = np.stack(X)
# test_input = X[:, 0:48]
# test_target = X[:, 48:96]
# test_name='MT_320'+"_test_input.txt"
# test_name2 ='MT_320' + "_test_target.txt"
# np.savetxt(test_name, test_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
# np.savetxt(test_name2, test_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
#
# print("finishing!")






