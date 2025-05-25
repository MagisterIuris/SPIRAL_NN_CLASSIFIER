import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt


class SPIRALNet(nn.Module): 
    def __init__(self): 
        super(SPIRALNet, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 1)
        self.activation = nn.Sigmoid()
        self.activation_cachee = nn.ReLU()

    def forward(self, entree): 
        entree = self.activation_cachee(self.layer1(entree))              
        entree = self.activation_cachee(self.layer2(entree))
        entree = self.activation(self.layer3(entree))
        return entree 
