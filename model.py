from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Lambda
from keras.optimizers import SGD

import keras.backend as K


def base_network(input_shape=(105,105,1)):

    input_image = Input(shape=input_shape)

    conv1 = Conv2D(filters=64,
                 kernel_size=(10,10),
                 activation='relu',
                 kernel_initializer='he_normal')(input_image)
    mp1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(filters=128,
                     kernel_size=(7,7),
                     activation='relu',
                     kernel_initializer='he_normal')(mp1)
    mp2 = MaxPooling2D()(conv2)
    conv3=Conv2D(filters=128,
                     kernel_size=(4,4),
                     activation='relu',
                     kernel_initializer='he_normal')(mp2)
    mp3 = MaxPooling2D()(conv3)
    conv4 = Conv2D(filters=256,
                     kernel_size=(4,4),
                     activation='relu',
                     kernel_initializer='he_normal')(mp3)
    flat1 = Flatten()(conv4)
    dense = Dense(units=4096,
                    activation='sigmoid',
                    name='Dense1')(flat1)


    return Model(input_image, dense)


def siamese_architecture(input_shape=(105,105,1), learning_rate=0.01):

    base_cnn = base_network(input_shape)

    input_image1 = Input(shape=input_shape)
    input_image2 = Input(shape=input_shape)

    image_rep1 = base_cnn(input_image1)
    image_rep2 = base_cnn(input_image2)

    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([image_rep1, image_rep2])

    pred = Dense(units=1,
                 activation='sigmoid')(l1_distance)

    model = Model(inputs=[input_image1, input_image2], outputs=pred)

    sgd = SGD(lr=learning_rate, momentum=0.9)

    model.compile(loss='binary_crossentropy',
                   metrics=['binary_accuracy'],
                   optimizer=sgd)

    return model