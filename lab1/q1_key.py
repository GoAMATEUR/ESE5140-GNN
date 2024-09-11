import numpy as np


def SampleGenerator(m,n):
    return np.random.binomial(1,1/(m),(m,n))


def InputsGenerator(n):
    return np.random.normal(np.zeros(n), ((np.sqrt(0.5 / n))) * np.ones(n))


def TrainingSetGenerator(A,q):
    [m,n]=A.shape
    X = InputsGenerator(n)
    Y = (A @ X + InputsGenerator(m))

    for i in range(q - 1):
        X = np.column_stack((X, InputsGenerator(n)))
        Y = np.column_stack((Y, A @ X[:, -1] + InputsGenerator(m)))

    X = X.T
    Y = Y.T
    return X,Y


# def TrainingSetGenerator(A,q):
#     [m,n]=A.shape
#     X = InputsGenerator(n)
#     Y = (A @ X + InputsGenerator(m))

#     for i in range(q - 1):
#         X = np.column_stack((X, InputsGenerator(n)))
#         Y = np.column_stack((Y, A @ X[:, -1] + InputsGenerator(m)))

#     X = X.T
#     Y = np.sign(Y.T)
#     return X,Y