import os

from keras import optimizers, regularizers
import keras
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
from keras.initializers import he_normal
from keras.layers import  Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dropout
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy

import keras.backend as K
import numpy as np
from operations import Convolution2D as Conv2D
from utills import load_cifar_100


#from keras.layers.convolutional import Conv2D
os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')

# from keras.layers import Conv2D



stack_n            = 5                  
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 240
iterations         = 50000 // batch_size
weight_decay       = 0.0001
# dropout = 0.5
mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

def scheduler(epoch):
    if epoch < 80:
        return 0.01
    if epoch < 160:
        return 0.001
    return 0.0001


k = 1

def build_model():
    model = Sequential()
    model.add(Conv2D(32*k, (5, 5), padding='valid', activation = 'relu',kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer='he_normal', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64*k, (5, 5), padding='valid', activation = 'relu',kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024*k, activation = 'relu',kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer='he_normal'))
    model.add(Dense(1024*k, activation = 'relu',kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer='he_normal'))
    model.add(Dense(100, activation = 'softmax',kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer='he_normal'))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model



def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

(x_train, y_train), (x_test, y_test) = load_cifar_100()

    
# color preprocessing
x_train, x_test = color_preprocessing(x_train, x_test)
#  
# build network
model = build_model()
print(model.summary())

# set callback
cbks = [TensorBoard(log_dir='./resnet_32/', histogram_freq=0),
        LearningRateScheduler(scheduler),
        ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)]

# set data augmentation
print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
datagen.fit(x_train)

# start training
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                     steps_per_epoch=iterations,
                     epochs=epochs,
                     callbacks=cbks,
                     validation_data=(x_test, y_test))

