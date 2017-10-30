import data_helper
import keras
import numpy as np
import defs

def interpret_result(yhati):
  """

  :param yhati: result of prediction for a file
  :return: String: the language
  """
  for i in range(0, len(yhati)):
    if yhati[i] > 0.5:
      return defs.langs[i]


# this is due to an unknown issue:
# https://stackoverflow.com/questions/47025036/keras-network-can-never-classify-the-last-class
defs.langs = defs.langs[0:15]

X, Y = data_helper.get_input_and_labels(data_helper.test_root_folder,
                                        defs.file_characters_truncation_limit,
                                        max_files=1000)

x = np.array(X)
model = keras.models.load_model('./save_tmp.h5')
y_hat = model.predict(x)

success = 0
class_success = {}
class_count = {}
for i in range(0, len(y_hat)):
  yi = Y[i]
  y_hati = y_hat[i]
  expected = interpret_result(yi)
  predicted = interpret_result(y_hati)

  if expected not in class_count:
    class_count[expected] = 1
  else:
    class_count[expected] += 1
  if expected == predicted:
    success += 1
    if predicted not in class_success:
      class_success[predicted] = 1
    else:
      class_success[predicted] += 1

print "Final result: {}/{} ({})".format(success, len(y_hat), (success * 1.0 / len(y_hat) * 1.0))

for key in class_count:
  if key not in class_success:
    class_success[key] = 0
  print "{}:\t\t{}/{} ({})".format(key, class_success[key], class_count[key], class_success[key] * 1.0 / class_count[key] * 1.0)

