import h5py
import numpy as np
np.random.seed(123)

# keras model module
from keras.models import Sequential
# keras core layers
from keras.layers import Dense, Dropout, Activation, Flatten
# keras CNN layers
from keras.layers import Conv2D, MaxPooling2D
# keras utils
from keras.utils import np_utils


# load MNIST dataset
from keras.datasets import mnist

# load preshuffled MNIST data into train and test sets
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

print Xtrain.shape

from matplotlib import pyplot as plt
# show image
# plt.imshow(Xtrain[0])
# plt.show()

# transform  dataset to having shape (n, depth, width, height).
Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, 28, 28)
Xtest = Xtest.reshape(Xtest.shape[0], 1, 28, 28)


print Xtrain.shape

# final preprocessing step for the input data is to convert our data type
# to float32
Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
# and normalize our data values to the range [0, 1].
Xtrain /= 255
Xtest /= 255


print Ytrain.shape
print Ytrain[:10]

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Ytrain = np_utils.to_categorical(Ytrain, 10)
Ytest = np_utils.to_categorical(Ytest, 10)

print Ytrain.shape

# Building model
model = Sequential()
# correspond to the number of convolution filters to use.
# the number of rows in each convolution kernel.
# and the number of columns in each convolution kernel.
# dim_ordering tells model to use Theano's dimension ordering
model.add(Conv2D(32, 3, 3, activation='relu',
                 input_shape=(1, 28, 28), dim_ordering='th'))
print model.output_shape

# add more layers to our model
model.add(Conv2D(32, 3, 3, activation='relu'))
# MaxPooling2D is a way to reduce the number of parameters in our model by
# sliding a 2x2 pooling filter across the previous layer and taking the
# max of the 4 values in the 2x2 filter
model.add(MaxPooling2D(pool_size=(2, 2)))
# regularizing our model in order to prevent overfitting
model.add(Dropout(0.25))
# note that the weights from the Convolution layers must be flattened
# (made 1-dimensional) before passing them to the fully connected Dense
# layer.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# final layer has an output size of 10, corresponding to the 10 classes of
# digits.
model.add(Dense(10, activation='softmax'))

# compile model
# https://keras.io/losses/
# https://keras.io/optimizers/
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train model
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(Xtest, Ytest, verbose=0)
print score

model.save("mnist.h5")
