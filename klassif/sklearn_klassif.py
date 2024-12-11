# Модель для тестирования взята с сайта: https://teletype.in/@pythontalk/sklearn_pytorch_nn

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm

# Загрузка и разделение датасета MNIST тренировочную и тестовую выборки
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X / 255.,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=1)

# # Сначала давайте используем очень простой персептрон;
# # Мы указываем гиперпараметры и обучаем персептрон, не используя
# # функции активации. Это означает, что модель по сути линейна.
# per = Perceptron(random_state=1,
#                  max_iter=30,
#                  tol=0.001)
# per.fit(X_train, y_train)
#
# # Делаем прогнозы с помощью построенного персептрона
# yhat_train_per = per.predict(X_train)
# yhat_test_per = per.predict(X_test)

# print(f"Перцептрон: точность на трейне: {accuracy_score(y_train, yhat_train_per)}")
# print(f"Перцептрон: точность на тесте  : {accuracy_score(y_test, yhat_test_per)}")


# Теперь давайте попробуем многослойный персептрон
# ReLU - функция активации по умолчанию

mlp = MLPClassifier(max_iter=50,
                    alpha=1e-4,
                    batch_size = 32,
                    solver='sgd',
                    verbose=0, #20,
                    random_state=1,
                    learning_rate_init=.1,
                    hidden_layer_sizes=(784, 100, 2))
def skl_klassif():

    res_iter = []

    # Начало работы классификации объектов
    t_start_lr = tm.time()

    mlp.fit(X_train, y_train)

    # Предсказываем результаты на тестовых данных
    y_pred = mlp.predict(X_test)

    # Оцениваем качество модели
    accuracy_kl = accuracy_score(y_test, y_pred)

    # Время работы классификации объектов
    t_work_lr = (tm.time() - t_start_lr) * 1000

    # Добавляем результаты работы в результирующую таблицк
    # acc = accuracy_score(accuracy_kl)
    # res_iter.append(acc)
    res_iter.append(accuracy_kl)
    res_iter.append(t_work_lr)

    return res_iter

def run_test(col_iter):
    res_func = []
    for i in range(col_iter):
        res_func.append(skl_klassif())
    arr = np.array(res_func)
    return arr

# print(run_test(1))
