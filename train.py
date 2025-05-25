import torch
import torch.nn as nn 
import torch.optim as optim
from model import SPIRALNet
from dataset import generate_spirals
import matplotlib.pyplot as plt

X_train, y_train = generate_spirals()
model = SPIRALNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.4)

loss_history = []
for epoch in range(100000):
    output = model(X_train)
    loss = criterion(output, y_train)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

with torch.no_grad():
    predictions = model(X_train)
    for i in range(4):
        print(f"Input: {X_train[i].tolist()} → Predicted: {predictions[i].item():.4f} (Target: {y_train[i].item()})")

torch.save(model.state_dict(), "saved_model/spiral_model.pth")