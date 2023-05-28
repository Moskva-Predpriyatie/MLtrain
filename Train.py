import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import torch
from tqdm.notebook import tqdm
import json
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import random
from sklearn.utils import shuffle
from IPython import display


def plot_loss(Loss_train=None, Loss_val=None):
    plt.figure(figsize=(12, 5))
    if Loss_val != None:
        plt.plot(range(len(Loss_val)), Loss_val, color='blue', marker='o', label='val')
    if Loss_train != None:
        plt.plot(range(len(Loss_train)), Loss_train, color='orange', label='train', linestyle='--')
    plt.legend()
    plt.show()


def train_main(model, opt, loss_fn, epochs, train_loader):
    for epoch in range(epochs):
        # печатаем номер текущей эпохи
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        # 1. Обучаем сеть на картинках из train_loader
        model.train()  # train mode

        avg_train_loss = 0
        losses = []
        for i, batch in enumerate(train_loader):
            # переносим батч на GPU
            X, Y = batch
            Y = Y.view(-1, 8).to(device)
            X = X.view(-1, 9).to(device)
            # получаем ответы сети на батч
            Y_pred = model(X)

            # считаем лосс, делаем шаг оптимизации сети
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())

            avg_train_loss += loss / len(train_loader)

        display.clear_output(wait=True)
        time.sleep(1)
        plot_loss(losses)
        display.display(plt.gcf())

        # выводим средний лосс на тренировочной выборке за эпоху
        print('avg train loss: %f' % avg_train_loss)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X = self.X.iloc[index].to_numpy()
        y = self.y.iloc[index].to_numpy()
        tensorX = torch.Tensor(X).view(1, 9)
        tensorY = torch.Tensor(y).view(1, 8)
        return tensorX, tensorY

    def __len__(self):
        return len(self.X)


class CustomFCL(nn.Module):
    def __init__(self):
        super(CustomFCL, self).__init__()

        self.len = 8
        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 200)
        self.fc5 = nn.Linear(200, self.len)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


np.random.seed(10000)
remove_n = 75000
X = pd.read_pickle('./X')
Y = pd.read_pickle('./Y')
drop_indices = np.random.choice(X.index, remove_n, replace=False)
X_ = X.drop(drop_indices)
Y_ = Y.drop(drop_indices)
print('check')
train_data = CustomDataset(X_, Y_)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'
custom_model = CustomFCL().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.0001)

train_main(custom_model, optimizer, criterion, 2, train_loader)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in custom_model.state_dict():
    print(param_tensor, "\t", custom_model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

torch.save(custom_model.state_dict(), './FCLmodelfinal')
