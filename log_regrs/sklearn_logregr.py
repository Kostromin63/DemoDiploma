# Импортируем необходимые библиотеки
import numpy as np
import time as tm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загружаем данные
X_train = np.load('make_classif/mk_cl_x_train.npy')
X_test = np.load('make_classif/mk_cl_x_test.npy')

y_train = np.load('make_classif/mk_cl_y_train.npy')
y_test = np.load('make_classif/mk_cl_y_test.npy')

def start_test():

    res_iter = []

    # Начало работы логистическое регрессии
    t_start_lr = tm.time()

    # Обучаем модель логистической регрессии
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Предсказываем результаты на тестовых данных
    y_pred = model.predict(X_test)

    # Оцениваем качество модели
    accuracy_lr = accuracy_score(y_test, y_pred)

    # Время работы лог. регрессии
    t_work_lr = (tm.time() - t_start_lr) * 1000

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