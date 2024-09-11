import numpy as np

def model_generator(n, m):
    """_summary_

    Args:
        n (int): input dim
        m (int): output dim

    Returns:
        np.ndarray: A (m, n)
    """
    
    return np.random.binomial(1, 1/m, (m, n))


def linear_sample(A: np.ndarray, Q: int):
    """

    Args:
        A (np.ndarray): A (m, n).
        Q (int): # of samples
    
    Returns:
        np.ndarray: x (Q, n)
        np.ndarray: y (Q, m)
    """
    m, n = A.shape[0], A.shape[1]
    x = np.random.normal(np.zeros((Q, n)), np.sqrt(1/2/n))
    w = np.random.normal(np.zeros((Q, m)), np.sqrt(1/2/m))
    y = x @ A.T + w
    return x, y


def sign_sample(A: np.ndarray, Q: int):
    """

    Args:
        A (np.ndarray): A (m, n).
        Q (int): # of samples
    
    Returns:
        np.ndarray: x (Q, n)
        np.ndarray: y (Q, m)
    """
    m, n = A.shape[0], A.shape[1]
    x = np.random.normal(np.zeros((Q, n)), np.sqrt(1/2/n))
    w = np.random.normal(np.zeros((Q, m)), np.sqrt(1/2/m))
    y = np.sign(x @ A.T + w)
    return x, y


if __name__ == "__main__":
    n = 10
    x1 = np.random.randn(10) * np.sqrt(1/2)
    x2 = np.random.normal(np.zeros((2, n)), np.sqrt(1/2/n))
    print(x1, x2)