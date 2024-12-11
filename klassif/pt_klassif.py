# Модель для анализа данных взята: https://github.com/selfedu-rus/neuro-pytorch/blob/main/neuro_net_21.py

# import os
# import json
# from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm
from torchvision.datasets import ImageFolder
import numpy as np
import time as tm


class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


model = DigitNN(28 * 28, 32, 10)

transforms = tfs.Compose([tfs.ToImage(),  tfs.Grayscale(),
                          tfs.ToDtype(torch.float32, scale=True),
                          RavelTransform(),
                          ])
d_train = ImageFolder("dataset/train", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

d_test = ImageFolder("dataset/test", transform=transforms)
test_data = data.DataLoader(d_test, batch_size=32, shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
epochs = 2
model.train()

def pt_klassif():
    res_iter = []

    # Начало работы классификации объектов
    t_start_lr = tm.time()

    for _e in range(epochs):
        # loss_mean = 0
        # lm_count = 0

    #    train_tqdm = tqdm(train_data, leave=True)
        train_tqdm = train_data
        for x_train, y_train in train_tqdm:
            predict = model(x_train)
            loss = loss_function(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lm_count += 1
            # loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
            # train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

    Q = 0

    # тестирование обученной НС
    model.eval()

    for x_test, y_test in test_data:
        with torch.no_grad():
            p = model(x_test)
            p = torch.argmax(p, dim=1)
            Q += torch.sum(p == y_test).item()

    Q /= len(d_test)
    # Время работы классификации объектов
    t_work_lr = (tm.time() - t_start_lr) * 1000

    # Добавляем результаты работы в результирующую таблицк
    res_iter.append(Q)
    res_iter.append(t_work_lr)

    return res_iter

def run_test(col_iter):
    res_func = []
    for i in range(col_iter):
        res_func.append(pt_klassif())
    arr = np.array(res_func)
    return arr

# print(run_test(1))
