import os
import codecs
import defs

root_folder = 'data/train'

def get_input_and_labels(file_vector_size=10 * 1024, max_files=1000):
  """


  :return: X, Y
  X is a an array of inputs where each input is a padded array of quantised character one-hot vectors
  Y is basically one-hot vector of the class definition
  """
  Y = []
  langs = defs.langs
  n_classes = len(langs)
  print n_classes
  X = []
  i = 0
  for fld in langs:
    vect = [0 for x in range(0, n_classes)]
    vect[i] = 1
    print fld
    folder = os.path.join(root_folder, fld)
    n = 0
    for fn in os.listdir(folder):
      if n > max_files:
        break
      file_name = os.path.join(folder, fn)
      try:
        file_vector = turn_file_to_vector(file_name, file_vector_size)
        X.append(file_vector)
        Y.append(vect)
        n += 1
      except Exception as e:
        print e
    i += 1

  return X, Y

def turn_file_to_vector(file_name, file_vector_size=10 * 1024):
  file_vector = []  # will be byte array
  text = ""
  with codecs.open(file_name, mode='r', encoding='utf-8') as f:
    text = f.read()
  text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
  text = text[0:file_vector_size]
  for ch in text:
    if ch in defs.supported_chars_map:
      file_vector.append(defs.supported_chars_map[ch])

  if len(file_vector) < file_vector_size:
    for j in range(0, file_vector_size - len(file_vector)):
      file_vector.append(defs.pad_vector)

  return file_vector

