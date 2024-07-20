import numpy as np
import pandas as pd
from os.path import join as pjoin
#####训练数据读取###################################
train_file = pjoin('./jn_data_2021', 'train_data_20190202_20201231.csv')
usecols = ['F_STATION_NAME','F_MONITOR_TIME','F_PM25']
df_data = pd.read_csv(train_file, usecols=usecols, error_bad_lines=False, low_memory=False,
                       encoding='utf-8', keep_default_na=False)
df_data = df_data.sort_values(by='F_STATION_NAME')

name_text = list(set(df_data['F_STATION_NAME'].tolist()))
num=0
X_min=0  #1.0
X_max=1  #392.0
for name, group in df_data.groupby('F_STATION_NAME'):
    df = group.sort_values(by='F_MONITOR_TIME')
    pm25 = pd.to_numeric(df['F_PM25'], errors='coerce')
    pm25=pm25.values
    X_min = np.min(pm25)
    X_max = np.max(pm25)
    print("min=", X_min)
    print("max=", X_max)
    pm25=(pm25 - X_min) / (X_max - X_min)
    X=[]
    for i in range(len(pm25)-144+1):
        temp=pm25[i:i+144]
        X.append(temp)
    N=round(len(X)*0.8)
    X = np.stack(X)
    train_input = X[0:N, 0:72]
    train_target = X[0:N, 72:144]
    val_input = X[N:len(X), 0:72]
    val_target = X[N:len(X), 72:144]
    train_name="JN72_train_input.txt"
    train_name2 = "JN72_train_target.txt"
    val_name ="JN72_val_input.txt"
    val_name2 = "JN72_val_target.txt"
    num+=1
    np.savetxt(train_name, train_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt(train_name2, train_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt(val_name, val_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt(val_name2, val_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    if num==1:
        break
print("finishing!")

#####测试数据读取###################################
test_file = pjoin('./jn_data_2021', 'test_data_20210101_20210506.csv')
usecols = ['F_STATION_NAME','F_MONITOR_TIME','F_PM25']
df_data = pd.read_csv(test_file, usecols=usecols, error_bad_lines=False, low_memory=False,
                       encoding='utf-8', keep_default_na=False)
df_data = df_data.sort_values(by='F_STATION_NAME')

name_text = list(set(df_data['F_STATION_NAME'].tolist()))
num=0
for name, group in df_data.groupby('F_STATION_NAME'):
    df = group.sort_values(by='F_MONITOR_TIME')
    pm25 = pd.to_numeric(df['F_PM25'], errors='coerce')
    pm25=pm25.values
    pm25 = (pm25 - X_min) / (X_max - X_min)
    X=[]
    for i in range(len(pm25)-144+1):
        temp=pm25[i:i+144]
        X.append(temp)
    X = np.stack(X)
    test_input = X[:, 0:72]
    test_target = X[:, 72:144]
    test_name="JN72_test_input.txt"
    test_name2 = "JN72_test_target.txt"
    num+=1
    np.savetxt(test_name, test_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt(test_name2, test_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    if num==1:
        break

print("finishing!")





