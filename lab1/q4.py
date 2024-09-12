from q1 import *
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

def SGD(X, Y, lr=0.005, epochs=200):
    losses = []
    H = np.random.randn(Y.shape[1], X.shape[1])
    print(H.shape)
    for epoch in range(epochs):
        grad = -(Y - X @ H.T).T @ X
        H = H - lr * grad
        loss = np.mean(0.5 * np.linalg.norm(Y - X @ H.T, axis=1)**2)
        
        print(f"epoch: {epoch}, loss: {loss}")
        losses.append(loss)
    return H, losses


def SGD_torch(X, Y, A, H, lr=0.005, epochs=200):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    H = torch.randn(Y.size(1), X.size(1), dtype=torch.float32)
    losses = []
    H.requires_grad = True
    optimizer = optim.SGD([H], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(0.5 * torch.norm(Y - X @ H.T, dim=1)**2)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")
        losses.append(loss.item())
    return H.detach().numpy(), losses

if __name__ == "__main__":
    X = np.load("X.npy")
    Y = np.load("Y.npy")
    A = np.load("A.npy")
    H = np.load("H.npy")
    print(H.shape)
    H_sgd, losses = SGD(X, Y)
    print(losses)
    print("H is close to optimal:", np.isclose(H, H_sgd, atol=1e-2).all())
    plt.plot(np.arange(len(losses)), losses, label="SGD", )
    plt.show()