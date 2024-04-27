# Databricks notebook source
# MAGIC %md
# MAGIC Define CNN Model

# COMMAND ----------

import torch
import torch.nn as nn

class RegularizedCNN(nn.Module):
    def __init__(self, num_classes, weight_decay=1e-5):
        super(RegularizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Define the L2 regularization penalty
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

    def l2_regularization_loss(self):
        l2_reg_loss = 0
        for param in self.parameters():
            l2_reg_loss += torch.norm(param, p=2) ** 2
        return 0.5 * self.weight_decay * l2_reg_loss


# COMMAND ----------

# MAGIC %md
# MAGIC Define Parameters

# COMMAND ----------

import torch.optim as optim

model = RegularizedCNN(num_classes=25)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
