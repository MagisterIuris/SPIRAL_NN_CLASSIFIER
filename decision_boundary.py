import torch
from model import SPIRALNet
import matplotlib.pyplot as plt
from dataset import generate_spirals


x_vals = torch.linspace(-5, 5, 300)
y_vals = torch.linspace(-5, 5, 300)
xx, yy = torch.meshgrid(x_vals, y_vals, indexing='ij')
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
model = SPIRALNet()
model.load_state_dict(torch.load("saved_model/spiral_model.pth")) 
model.eval()

with torch.no_grad():
    probs = model(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, levels=100, cmap="coolwarm", alpha=0.8)
plt.colorbar(label="Probabilité classe 1")

X_visu, y_visu = generate_spirals(n_points=1000, noise=0.2)
plt.scatter(X_visu[:, 0], X_visu[:, 1], c=y_visu.squeeze(), cmap="coolwarm", s=10, edgecolors='k')

plt.title("Heatmap continue des prédictions du modèle (spirales)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
