import torch 
from model import SPIRALNet
from dataset import generate_spirals
import matplotlib.pyplot as plt


model = SPIRALNet()
model.load_state_dict(torch.load("saved_model/spiral_model.pth"))
model.eval()

X_test, y_test = generate_spirals(n_points=1000, noise=0.2)

with torch.no_grad():
    preds = model(X_test)
    preds_bin = (preds > 0.5).float()

accuracy = (preds_bin == y_test).float().mean().item()
print(f"Accuracy on test spirals: {accuracy * 100:.2f}%")

plt.figure(figsize=(6, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=preds_bin.squeeze(), cmap="coolwarm", s=15, edgecolors='k')
plt.title("Model Predictions on Spiral Test Set")
plt.axis("equal")
plt.grid(True)
plt.show()

