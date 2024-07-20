import numpy as np
import pandas as pd
from os.path import join as pjoin
#####训练数据读取###################################
train_file = pjoin('./ETT_data/ETT-small', 'ETTh2.csv')
usecols = ['date','HUFL']
df_data = pd.read_csv(train_file, usecols=usecols, error_bad_lines=False, low_memory=False,
                       encoding='utf-8', keep_default_na=False)
#df_data = df_data.sort_values(by='date')
ett= pd.to_numeric(df_data['HUFL'], errors='coerce')
ett_v=ett.values
length = len(ett_v)
# #------zero-mean normalization---
# x_mean = np.mean(ecl_v).astype(float)
# vari = np.sqrt((np.sum((ecl_v-x_mean)**2))/length)
# ecl_norm = (ecl_v-x_mean)/vari

#Linear normalization--Max-Min
X_min = np.min(ett_v)
X_max = np.max(ett_v)
## x_max= 107.89299774169922
## x_min=0
print("min=",X_min)
print("max=",X_max)
ett_norm = (ett_v - X_min) / (X_max - X_min)
np.any(np.isnan(ett_norm))
nb=round(length*0.8)
ett_train=ett_norm[0:nb]
ett_test=ett_norm[nb:]

# nb=round(length*0.82)
# ecl_train=ecl_v[0:nb]
# ecl_test=ecl_v[nb:]

X=[]
for i in range(len(ett_train) - 144 + 1):
    temp = ett_train[i:i + 144]
    X.append(temp)
N=round(len(X)*0.75)
X = np.stack(X)
train_input = X[0:N, 0:72]
train_target = X[0:N, 72:144]
val_input = X[N:len(X), 0:72]
val_target = X[N:len(X), 72:144]
train_name='ETT72'+"_train_input.txt"
train_name2 = 'ETT72' + "_train_target.txt"
val_name = 'ETT72' + "_val_input.txt"
val_name2 = 'ETT72' + "_val_target.txt"
np.savetxt(train_name, train_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(train_name2, train_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(val_name, val_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(val_name2, val_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔

Y=[]
for i in range(len(ett_test) - 144 + 1):
    temp = ett_test[i:i + 144]
    Y.append(temp)
Y = np.stack(Y)
test_input = Y[:, 0:72]
test_target = Y[:, 72:144]
test_name='ETT72'+"_test_input.txt"
test_name2 ='ETT72' + "_test_target.txt"
np.savetxt(test_name, test_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
np.savetxt(test_name2, test_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔

print("finishing!")
