import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time as tm
from tensorflow import keras
from tensorflow.keras import layers

# отключаем прогресс-бар
tf.keras.utils.disable_interactive_logging()

# Загружаем данные
X_train = np.load('make_classif/mk_cl_x_train.npy')
X_test = np.load('make_classif/mk_cl_x_test.npy')

y_train = np.load('make_classif/mk_cl_y_train.npy')
y_test = np.load('make_classif/mk_cl_y_test.npy')

# Отделите целевое значение — «метку» — от признаков. Эта метка является значением,
# которое вы будете обучать модели прогнозировать.
train_features = X_train.copy()
test_features = X_test.copy()

# Первым шагом является создание слоя:
normalizer = tf.keras.layers.Normalization(axis=-1)

# Затем подгоните состояние слоя предварительной обработки к данным, вызвав Normalization.adapt :
normalizer.adapt(np.array(train_features))

# Вычислите среднее значение и дисперсию и сохраните их в слое:
normalizer.mean.numpy()

#Когда слой вызывается, он возвращает входные данные, причем каждая функция нормализована независимо:
first = np.array(train_features[:1])

def start_test():
    res_iter = []

    # Начало работы логистическое регрессии
    t_start_lr = tm.time()

    # Линейная регрессия
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )

    x = tf.ones((3, 3))
    y = model(x)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.Accuracy(),
        ],
    )
    model.fit(x, y, batch_size=32, epochs=10)

    accuracy_lr = model.get_metrics_result()['accuracy']
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

if __name__ == '__main__':
    print(run_test(2))
