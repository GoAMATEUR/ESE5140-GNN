import numpy as np
from q1 import *
# from matplotlib import pyplot as plt

def l_mse(H, x, y):
    """_summary_
    Args:
        H (_type_): (m, n)
        x (_type_): (Q, n)
        y (_type_): (Q, m)
    """
    y_hat = x @ H.T
    return np.mean(0.5 * np.linalg.norm(y - y_hat, axis=1)**2)

if __name__ == "__main__":
    n = m = 100
    Q = 1000

    A = model_generator(n, m)
    
    X, Y = linear_sample(A, Q)
    X_, Y_ = linear_sample(A, Q)
    H_opt = Y.T @ X @ np.linalg.inv(X.T @ X)
    
    loss = l_mse(H_opt, X, Y)
    loss_eval = l_mse(H_opt, X_, Y_)
    
    np.save("X.npy", X)
    np.save("Y.npy", Y)
    np.save("A.npy", A)
    np.save("H.npy", H_opt)
    
    print("loss linear: ", loss)
    print("loss linear eval: ", loss_eval)
    
    # X, Y = sign_sample(A, Q)
    # X_, Y_ = sign_sample(A, Q)
    # H_opt = Y.T @ X @ np.linalg.inv(X.T @ X)
    # loss = l_mse(H_opt, X, Y)
    # loss_eval = l_mse(H_opt, X_, Y_)
    # print("loss sign: ", loss)
    # print("loss sign eval: ", loss_eval)