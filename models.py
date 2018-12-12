# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAvgPool2D
from keras.models import Model
from keras import optimizers
from keras.utils import plot_model


def CNN(input_shape=(60,65,1), nclass=10):
    """
    build a simple cnn model using keras with TensorFlow backend.
    :param input_shape: input shape of network, default as (60,65,1)
    :param nclass: numbers of class(output shape of network), default as 10
    :return: cnn model
    """
    input_ = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_ = Dense(nclass, activation='softmax')(x)

    model = Model(inputs=input_, outputs=output_)
    # 输出模型的参数信息
    model.summary()
    # 配置模型训练过程
    sgd = optimizers.sgd(lr=0.01, momentum=0.9, nesterov=True)  # 优化器为SGD
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # 交叉熵为cross entropy

    return model

if __name__ == '__main__':
    model = CNN()
    plot_model(model, './image/cnn.png')  # 保存模型图