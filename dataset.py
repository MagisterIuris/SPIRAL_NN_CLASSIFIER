import torch
import math
import matplotlib.pyplot as plt

def generate_spirals(n_points=1000, noise=0.2):
    n_class = n_points // 2
    X = []
    y = []

    for i in range(n_class):
        r = i / n_class * 5
        t = 1.75 * i / n_class * 2 * math.pi

        x1 = r * math.sin(t) + torch.randn(1).item() * noise
        y1 = r * math.cos(t) + torch.randn(1).item() * noise
        X.append([x1, y1])
        y.append(0)

        x2 = r * math.sin(t + math.pi) + torch.randn(1).item() * noise
        y2 = r * math.cos(t + math.pi) + torch.randn(1).item() * noise
        X.append([x2, y2])
        y.append(1)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y

if __name__ == "__main__":
    X, y = generate_spirals()

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='coolwarm', s=10)
    plt.title("Spirales imbriquées")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
