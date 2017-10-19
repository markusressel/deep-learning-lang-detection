import data_helper
import keras
import sys
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


X, Y = data_helper.get_input_and_labels(data_helper.test_root_folder,
                                        defs.file_characters_truncation_limit,
                                        max_files=100)

x = np.array(X)
model = keras.models.load_model('./save_tmp.h5')
y_hat = model.predict(x)

success = 0
for i in range(0, len(y_hat)):
  yi = Y[i]
  y_hati = y_hat[i]
  expected = interpret_result(yi)
  predicted = interpret_result(y_hati)

  if expected == predicted:
    success += 1

print "Final result: {}/{} ({})".format(success, len(y_hat), (success * 1.0 / len(y_hat) * 1.0))


