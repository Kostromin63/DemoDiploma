import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import time as tm
import keras


class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b, trainable=False)

            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == "relu":
            return tf.nn.relu(y)
        elif self.activate == "softmax":
            return tf.nn.softmax(y)

        return y


class SequentialModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseNN(128)
        self.layer_2 = DenseNN(10, activate="softmax")

    def __call__(self, x):
        return self.layer_2(self.layer_1(x))

# Так как  готовый датасет выгрузить на гитхаб нет технической возможности, ограничение по размеру файлов до 100 мб
# я закомментировал следующий код:
# x_train = np.load('npy/mnist_x_train.npy')
# y_train = np.load('npy/mnist_y_train.npy')
#
# x_test = np.load('npy/mnist_x_test.npy')
# y_test = np.load('npy/mnist_y_test.npy')

# Будем загружать данные из библиотеки keras и приводить к нужному нам виду.
mnist = keras.datasets.mnist
to_categorical = keras.utils.to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_train, 10)

model = SequentialModule()

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.optimizers.Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 20
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

def tf_klassif():
    res_iter = []

    # Начало работы классификации объектов
    t_start_lr = tm.time()

    @tf.function
    def train_batch(x_batch, y_batch):
        with tf.GradientTape() as tape:
            f_loss = cross_entropy(y_batch, model(x_batch))

        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        return f_loss

    for n in range(EPOCHS):
        loss = 0
        for x_batch, y_batch in train_dataset:
            loss += train_batch(x_batch, y_batch)

    acc = tf.metrics.Accuracy()

    # Время работы классификации объектов
    t_work_lr = (tm.time() - t_start_lr) * 1000

    # Добавляем результаты работы в результирующую таблицк
    res_iter.append(acc.result().numpy() * 100)
    res_iter.append(t_work_lr)

    return res_iter

def run_test(col_iter):
    res_func = []
    for i in range(col_iter):
        res_func.append(tf_klassif())
    arr = np.array(res_func)
    return arr

if __name__ == '__main__':
    print(run_test(2))
