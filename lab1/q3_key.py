import numpy as np
from q1_key import *
# from matplotlib import pyplot as plt

def l_mse(H, x, y):
    """_summary_
    Args:
        H (_type_): (m, n)
        x (_type_): (Q, n)
        y (_type_): (Q, m)
    """
    y_hat = x @ H.T
    
    return np.mean(np.linalg.norm(y - y_hat, axis=1)**2)

if __name__ == "__main__":
    n = m = 100
    Q = 1000

    A = SampleGenerator(n, m)
    X, Y = TrainingSetGenerator(A, Q)

    H_opt = Y.T @ X @ np.linalg.inv(X.T @ X)
    loss = l_mse(H_opt, X, Y)
    print("loss linear: ", loss)
    
    

    
