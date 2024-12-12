import time as tm
from klassif import sklearn_klassif, pt_klassif, tf_klassif

application_queue = [sklearn_klassif, pt_klassif, tf_klassif]
comparison_table = []

number_of_line_in_the_array = 2

print(f'{number_of_line_in_the_array} циклов работы шести функций запущены. Это может занять продолжительное время, '
      f'до 10 минут на один цикл.')
t_start = tm.time()
for app in application_queue:
    comparison_table.append(app.run_test(number_of_line_in_the_array))
t_work = tm.time() - t_start

for arr in comparison_table:
    print(arr)
