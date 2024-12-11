import torch
import torch.nn as nn
import torch.optim as optim
import time as tm
import numpy as np
from sklearn.metrics import classification_report

# Загружаем данные
X = np.load('make_classif/mk_cl_x.npy')
y = np.load('make_classif/mk_cl_y.npy')

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x @ self.weights
        x = self.sigmoid(x)
        return x

    def fit(self, X, y, lr=0.01, num_iterations=1000):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float().view(-1, 1)

        # Инициализируем функцию потерь и оптимизатор
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(num_iterations):
            # Зануляем градиенты
            optimizer.zero_grad()

            # Получаем предсказания модели и вычисляем функцию потерь
            y_pred = self(X)
            loss = criterion(y_pred, y)

            # Обновляем веса
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X = torch.from_numpy(X).float()

        # Получаем предсказания модели и присваиваем метки классов на основе вероятности
        y_pred = self(X)
        y_pred_labels = [1 if i > 0.5 else 0 for i in y_pred.detach().numpy().flatten()]

        return y_pred_labels


def start_test():
    res_iter = []

    # Начало работы логистическое регрессии
    t_start_lr = tm.time()
    # Создаем экземпляр класса и обучаем на обучающей выборке
    model = LogisticRegression(X.shape[1])
    model.fit(X, y, lr=0.1, num_iterations=100)

    # Прогнозируем метки классов на тестовой выборке
    y_pred = model.predict(X)

    # Время работы лог. регрессии
    t_work_lr = (tm.time() - t_start_lr) * 1000
    accuracy_lr = classification_report(y, y_pred, output_dict=True)['accuracy']
    # Добавляем результаты работы в результирующую таблицк
    res_iter.append(accuracy_lr)
    res_iter.append(t_work_lr)
    return res_iter

def run_test(col_iter):
    res_func = []
    for i in range(col_iter):
        res_func.append(start_test())
        # print(start_test())
    arr = np.array(res_func)
    return arr

run_test(4)
