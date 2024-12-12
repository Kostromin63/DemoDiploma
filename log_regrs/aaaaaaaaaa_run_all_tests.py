
from log_regrs import sklearn_logregr, pt_logregr, tf_logregr

application_queue = [sklearn_logregr, pt_logregr, tf_logregr]
comparison_table = []

number_of_line_in_the_array = 2

for app in application_queue:
    comparison_table.append(app.run_test(number_of_line_in_the_array))

for arr in comparison_table:
    print(arr)
