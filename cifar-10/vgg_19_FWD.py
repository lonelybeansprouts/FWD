import os

from keras import optimizers, regularizers
import keras
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
from keras.initializers import he_normal
from keras.layers import  Dense, Input, add, Activation, GlobalAveragePooling2D

from keras.layers.core import Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy

import keras.backend as K
import numpy as np

#from keras.layers.convolutional import Conv2D
from operations import Convolution2D as Conv2D

from utills import load_cifar
from keras.layers.merge import Concatenate


os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')

# from keras.layers import Conv2D



stack_n            = 5                  
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 50000 // batch_size
weight_decay       = 0.0001
dropout = 0.5
mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 150:
        return 0.01    
    return 0.001

# he_normal = truncated_normal

# build model
input = Input(shape=[32,32,3])


# Block 1
x = Conv2D(8, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=[32,32,3])(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# model modification for cifar-10
x = Flatten(name='flatten')(x)
x = Dense(1024, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(dropout)(x)
x = Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2')(x)  
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(dropout)(x)      
x = Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10')(x)        
x = BatchNormalization()(x)
output = Activation('softmax')(x)

model = Model(input,output)


def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

if __name__ == '__main__':
#     # load data
# #     (x_train, y_train), (x_test, y_test) = cifar10.load_data(r'F:\AAA_workspace\dataset\cifar-10-batches-py')
# #     y_train = keras.utils.to_categorical(y_train, num_classes)
# #     y_test = keras.utils.to_categorical(y_test, num_classes)
#  




   
    (x_train, y_train), (x_test, y_test) = load_cifar()
#     
# #     # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)
#  
    # build network
     
    print(model.summary())
  
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
#     sgd = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  
    # set callback
    cbks = [TensorBoard(log_dir='./resnet_32/', histogram_freq=0),
            LearningRateScheduler(scheduler),
            ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, save_weights_only=True, mode='auto', period=10)]
  
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

#     resnet = load_model('resnet_2.h5')
# #     r = resnet.evaluate(x_test, y_test,1)
#     r = resnet.predict(x_test, 100)
#     r = numpy.argmax(r,1)==numpy.argmax(y_test,1)
#     r = numpy.mean(r.astype('float32'))
#     print(r)


    
    