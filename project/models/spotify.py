import tensorflow as tf
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model


def get(input_shape, num_classes, batchnorm=True, activation='softmax'):
    """
    Based on "Recommending music on Spotify with Deep Learning"
    """
    input_layer = Input(shape=input_shape)
    layer = input_layer
    for i in range(3):
        layer = Conv1D(filters=256, kernel_size=4,
                       strides=2, padding='same',
                       kernel_initializer='glorot_normal')(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
    avg_pool = AveragePooling1D(pool_size=4)(layer)
    max_pool = MaxPooling1D(pool_size=4)(layer)
    layer = concatenate([avg_pool, max_pool])
    layer = Flatten()(layer)
    layer = Dense(units=2048, activation='relu')(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = Dense(units=num_classes)(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = Activation(activation)(layer)
    return Model(inputs=input_layer, outputs=layer)


def get_tf(x, num_classes, batchnorm=True, activation='softmax'):
    layer = x
    for i in range(3):
        with tf.variable_scope('conv_relu_pool_%d' % i):
            layer = tf.layers.conv1d(
                inputs=layer,
                filters=256,
                kernel_size=4,
                padding='same',
                activation=tf.nn.relu)
            layer = tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=2)
    avg_pool = tf.layers.average_pooling1d(inputs=layer, pool_size=2, strides=2)
    max_pool = tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=2)
    layer = tf.concat([avg_pool, max_pool], axis=-1)
    layer = tf.contrib.layers.flatten(layer)
    layer = tf.layers.dense(layer, units=2048)
    layer = tf.layers.dropout(layer, rate=0.5)
    layer = tf.layers.dense(layer, units=num_classes)
    layer = tf.layers.dropout(layer, rate=0.5)
    return layer

