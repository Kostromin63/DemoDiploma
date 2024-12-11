# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
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
# # create_make_classification()
