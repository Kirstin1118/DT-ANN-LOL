# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 23:15:56 2020

@author: Kirstin YU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

train_data = open(r'new_data.csv', 'r')
train = pd.read_csv(train_data)
train_data.close()

X = train.iloc[:, 5:]
X = torch.from_numpy(np.array(X))
y = train['winner']
y = torch.from_numpy(np.array(y - 1))
X_train = torch.as_tensor(X.clone().detach(), dtype=torch.float)
y_train = torch.as_tensor(y.clone().detach(), dtype=torch.long)

test_data = open(r'test_set.csv', 'r')
test = pd.read_csv(test_data)
test_data.close()

X = test.iloc[:, 5:]
X = torch.from_numpy(np.array(X))
y = test['winner']
y = torch.from_numpy(np.array(y - 1))
X_test = torch.as_tensor(X.clone().detach(), dtype=torch.float)
y_test = torch.as_tensor(y.clone().detach(), dtype=torch.long)

train_set = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=50, shuffle=True)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.output(x)
        x = F.softmax(x)
        return x


model = ANN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X_train)
    loss = criterion(y_hat, y_train)
    loss_arr.append(loss)

    if i % 2 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predict_out = model(X_test)

_, predict_y = torch.max(predict_out, 1)

predict_y

from sklearn.metrics import accuracy_score

print("The accuracy is ", accuracy_score(y_test, predict_y))
