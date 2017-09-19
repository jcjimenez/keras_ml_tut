from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import mnist

# load preshuffled MNIST data into train and test sets
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtest = Xtest.reshape(Xtest.shape[0], 1, 28, 28)
Xtest = Xtest.astype('float32')
Xtest /= 255
Ytest = np_utils.to_categorical(Ytest, 10)

model = load_model("mnist.h5")
score = model.evaluate(Xtest, Ytest, verbose=0)
print score
