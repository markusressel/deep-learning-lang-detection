import numpy as np
import data_helper
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
import defs

np.random.seed(2)

# data parameters
n_max_files = 1000 # max number of files per class to load
number_of_classes = len(defs.langs)

# Model Hyperparameters
sequence_length = defs.file_characters_truncation_limit
filter_sizes = (2, 5, 9, 19)
pooling_sizes = (2, 5, 9, 19)
num_filters = 128
dropout_prob = (0.3, 0.6)
hidden_dims = 128
number_of_quantised_characters = len(defs.supported_chars)


# Training parameters
batch_size = 100
num_epochs = 30
val_split = 0.1


# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")
x, y = data_helper.get_input_and_labels(file_vector_size=sequence_length, max_files=n_max_files)

# Setting up the model
graph_in = Input(shape=(sequence_length, number_of_quantised_characters))
convs = []
for i in range(0, len(filter_sizes)):
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=filter_sizes[i],
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=pooling_sizes[i])(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes)>1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)

# main sequential model
model = Sequential()

model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, number_of_quantised_characters)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Dense(number_of_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Training model
# ==================================================
model.fit(x, y, batch_size=batch_size,
          nb_epoch=num_epochs, validation_split=val_split, verbose=1)

model.save('save_tmp.h5')
