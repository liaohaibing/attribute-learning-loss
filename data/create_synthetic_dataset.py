import numpy as np
import random

def create_synthetic_dataset(N, N_input,N_output,sigma):
    # N: number of samples in each split (train, test)  500
    # N_input: import of time steps in input series 20
    # N_output: import of time steps in output series 20
    # sigma: standard deviation of additional noise 0.01
    X = []
    breakpoints = []
    for k in range(3*N):
        #random.seed(k)
        serie = np.array([ sigma*random.random() for i in range(N_input+N_output)])
        # serie = np.array([sigma * random.randrange(5, 500) for i in range(N_input + N_output)])
        i1 = random.randint(1,36)
        i2 = random.randint(36,70)
        j1 = random.random()
        j2 = random.random()
        interval = abs(i2-i1) + random.randint(-3,3)
        serie[i1:i1+1] += j1
        serie[i2:i2+1] += j2
        serie[i2+interval:] += abs(j2-j1)
        X.append(serie)
        breakpoints.append(i2+interval)
    X = np.stack(X)
    breakpoints = np.array(breakpoints)
    train_input= X[0:N,0:N_input]
    train_target = X[0:N, N_input:N_input+N_output]
    val_input = X[N:2*N,0:N_input]
    val_target = X[N:2*N, N_input:N_input+N_output]
    test_input = X[2*N:3*N, 0:N_input]
    test_target = X[2*N:3*N, N_input:N_input + N_output]
    train_bkp = breakpoints[0:N]
    val_bkp = breakpoints[N:2*N]
    test_bkp = breakpoints[2*N:3*N]
    np.savetxt("SY_train_input72.txt", train_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_train_target72.txt", train_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_val_input72.txt", val_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_val_target72.txt", val_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_test_input72.txt", test_input, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_test_target72.txt", test_target, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_train_bkp72.txt", train_bkp, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_val_bkp72.txt", val_bkp, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔
    np.savetxt("SY_test_bkp72.txt", test_bkp, fmt="%f", delimiter=",")  # 保存为浮点数，以逗号分隔

if __name__ == "__main__":
    create_synthetic_dataset(500, 72, 72, 0.01)