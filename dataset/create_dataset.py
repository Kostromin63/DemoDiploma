# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# os.environ["KERAS_BACKEND"] = "tensorflow"
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# import keras
# import tensorflow as tf
# import numpy as np
#
#
# def create_mnist():
#     mnist = keras.datasets.mnist
#     to_categorical = keras.utils.to_categorical
#
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#     x_train = x_train / 255
#     x_test = x_test / 255
#
#     x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
#     x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])
#
#     y_train = to_categorical(y_train, 10)
#     y_test = to_categorical(y_train, 10)
#
#     # Сохраняем себе в проект
#     np.save('mnist/npy/mnist_x_train',x_train)
#     np.save('mnist/npy/mnist_y_train',y_train)
#     np.save('mnist/npy/mnist_x_test',x_test)
#     np.save('mnist/npy/mnist_y_test', y_test)
#
#
# def create_make_classification():
#     # Генерируем данные
#     x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
#                                random_state=1, n_clusters_per_class=1)
#
#     # Разделяем данные на обучающую и тестовую части
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#
#     # Сохраняем себе в проект
#     np.save('make_classif/mk_cl_x', x)
#     np.save('make_classif/mk_cl_y', y)
#     np.save('make_classif/mk_cl_x_train', x_train)
#     np.save('make_classif/mk_cl_y_train', y_train)
#     np.save('make_classif/mk_cl_x_test', x_test)
#     np.save('make_classif/mk_cl_y_test', y_test)
#
#
# ## bloc create funcs
# #____________________________________________________
#
# # create_mnist()
# # create_make_classification()