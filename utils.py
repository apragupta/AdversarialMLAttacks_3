import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
def preprocess_mnist(x,y):
    x = x.astype("float32") / 255
    x = np.expand_dims(x,-1)
    y = keras.utils.to_categorical(y, 10)
    return x,y