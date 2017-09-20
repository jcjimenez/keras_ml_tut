import cv2
from keras.models import load_model
# from keras.utils import np_utils
# from keras.datasets import mnist
from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("mnist.h5")

# read in image
img = image.load_img("./images/9-v.png")
plt.imshow(img)
# plt.show()

# show dimensions
x = image.img_to_array(img)
print x.shape

# convert to 1 channel gray scale
x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
print x.shape

# convert to 4 dimensions: (nb_samples, nb_channels/depth, width, height)
x = x.reshape(1, 1, 28, 28)
# x = np.expand_dims(x, axis=0)
# x = np.reshape(x, (1, 28, 28, 1))
print x.shape

# predict
prediction = model.predict(x)
print np.around(prediction[0], decimals=0)
