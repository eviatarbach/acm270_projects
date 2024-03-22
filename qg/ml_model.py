import keras
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.callbacks import History
history = History()

from keras.models import Model

__version__ = 0.1

def stn(input_shape=(192, 96,2), sampling_size=(8, 16), num_classes=10):
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv3)

    conv5 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv5)

    up6 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x)), conv2])
    conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv6)

    up7 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv1])
    conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv7)

    conv10 = Convolution2D(2, (5, 5), activation='linear',padding='same')(conv7)

    model = Model(inputs, conv10)

    return model

model = stn()
model.compile(loss='mse', optimizer='adam')
model.summary()

model.load_weights('weights.h5')