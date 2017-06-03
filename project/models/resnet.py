"""
Implementation of Deep Residual Network.

References:
  [1] "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
  [2] "Identity Mappings in Deep Residual Networks" https://arxiv.org/pdf/1603.05027.pdf
"""
from keras import backend as K
from keras.layers import Activation, Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model


class ResidualUnit:
    """
    Residual unit as described in [1].
    """
    def __init__(self, filters, first_conv_strides):
        self.filters = filters
        self.first_conv_strides = first_conv_strides

    def __call__(self, x):
        conv1 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=self.first_conv_strides, padding='same',
                       kernel_initializer='glorot_normal')(x)
        norm1 = BatchNormalization(axis=3)(conv1)
        relu1 = Activation('relu')(norm1)
        conv2 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu1)
        norm2 = BatchNormalization(axis=3)(conv2)
        return Activation('relu')(self.shortcut_and_add(x, norm2))

    def shortcut_and_add(self, x, residual):
        x_shape = K.int_shape(x)
        residual_shape = K.int_shape(residual)
        shortcut = x
        if x_shape != residual_shape:
            conv1 = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                           strides=self.first_conv_strides, padding='same',
                           kernel_initializer='glorot_normal')(x)
            shortcut = BatchNormalization(axis=3)(conv1)
        return add([shortcut, residual])


class BottleneckResidualUnit(ResidualUnit):
    """
    Bottleneck residual unit as described in [1] for ResNet-50/101/152.
    """

    def __call__(self, x):
        conv1 = Conv2D(filters=self.filters, kernel_size=(1, 1),
                       strides=self.first_conv_strides, padding='same',
                       kernel_initializer='glorot_normal')(x)
        norm1 = BatchNormalization(axis=3)(conv1)
        relu1 = Activation('relu')(norm1)
        conv2 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu1)
        norm2 = BatchNormalization(axis=3)(conv2)
        relu2 = Activation('relu')(norm2)
        conv3 = Conv2D(filters=self.filters * 4, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu2)
        norm3 = BatchNormalization(axis=3)(conv3)
        return Activation('relu')(self.shortcut_and_add(x, norm3))


class IdentityResidualUnit(ResidualUnit):
    """
    Basic residual unit as described in [2].
    """
    def __call__(self, x):
        norm1 = BatchNormalization(axis=3)(x)
        relu1 = Activation('relu')(norm1)
        conv1 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=self.first_conv_strides, padding='same',
                       kernel_initializer='glorot_normal')(relu1)
        norm2 = BatchNormalization(axis=3)(conv1)
        relu2 = Activation('relu')(norm2)
        conv2 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu2)
        return self.shortcut_and_add(x, conv2)


class BottleneckIdentityResidualUnit(ResidualUnit):
    """
    Basic residual unit as described in [2].
    """
    def __call__(self, x):
        norm1 = BatchNormalization(axis=3)(x)
        relu1 = Activation('relu')(norm1)
        conv1 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=self.first_conv_strides, padding='same',
                       kernel_initializer='glorot_normal')(relu1)
        norm2 = BatchNormalization(axis=3)(conv1)
        relu2 = Activation('relu')(norm2)
        conv2 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu2)
        norm3 = BatchNormalization(axis=3)(conv2)
        relu3 = Activation('relu')(norm3)
        conv3 = Conv2D(filters=self.filters * 4, kernel_size=(1, 1),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu3)
        return self.shortcut_and_add(x, conv3)


class ResidualBlock:
    def __init__(self, units, filters, residual_unit_cls, is_first_block=False):
        self.filters = filters
        self.units = units
        self.is_first_block = is_first_block
        self.residual_unit_cls = residual_unit_cls

    def __call__(self, x):
        current = x
        for i in range(self.units):
            strides = (1, 1)
            if not self.is_first_block and i == 0:
                strides = (2, 2)
            current = self.residual_unit_cls(
                filters=self.filters, first_conv_strides=strides)(current)
        return current


class WideResidualUnit:
    # TODO
    pass


def get(input_shape, num_classes, residual_unit_cls, units_per_block):
    """As described in [1]"""
    x = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7, 7),
                   strides=(2, 2), padding='same',
                   kernel_initializer='glorot_normal')(x)
    norm1 = BatchNormalization(axis=3)(conv1)
    relu1 = Activation('relu')(norm1)
    current = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(relu1)
    filters = 64
    for i, units in enumerate(units_per_block):
        current = ResidualBlock(units, filters, residual_unit_cls, is_first_block=(i == 0))(current)
        filters *= 2
    relu1 = Activation('relu')(current)
    avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(relu1)
    flatten1 = Flatten()(avg_pool)
    dense = Dense(units=num_classes, activation='softmax')(flatten1)
    return Model(inputs=x, outputs=dense)


def get_18(input_shape, num_classes, unit_cls=ResidualUnit):
    """As described in [1]"""
    _validate_non_bottleneck_unit(unit_cls)
    return get(input_shape, num_classes, unit_cls, [2, 2, 2, 2])


def get_34(input_shape, num_classes, unit_cls=ResidualUnit):
    """As described in [1]"""
    _validate_non_bottleneck_unit(unit_cls)
    return get(input_shape, num_classes, unit_cls, [3, 4, 6, 3])


def get_50(input_shape, num_classes, unit_cls=BottleneckResidualUnit):
    """As described in [1]"""
    _validate_bottleneck_unit(unit_cls)
    return get(input_shape, num_classes, unit_cls, [3, 4, 6, 3])


def get_101(input_shape, num_classes, unit_cls=BottleneckResidualUnit):
    """As described in [1]"""
    _validate_bottleneck_unit(unit_cls)
    return get(input_shape, num_classes, unit_cls, [3, 4, 23, 3])


def get_152(input_shape, num_classes, unit_cls=BottleneckResidualUnit):
    """As described in [1]"""
    _validate_bottleneck_unit(unit_cls)
    return get(input_shape, num_classes, unit_cls, [3, 8, 36, 3])


def _validate_non_bottleneck_unit(unit_cls):
    if unit_cls not in (ResidualUnit, IdentityResidualUnit):
        raise ValueError('Invalid non bottleneck unit')


def _validate_bottleneck_unit(unit_cls):
    if unit_cls not in (BottleneckResidualUnit, BottleneckIdentityResidualUnit):
        raise ValueError('Invalid bottleneck unit')



def get_tf(x, num_classes):
    units_per_block = [2, 2, 2, 2]
    with tf.variable_scope('resnet'):
        layer = x
        layer = tf.layers.conv2d(
            inputs=layer,
            filters=conv_filter,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same')
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer)
        layer = tf.layers.max_pooling2d(layer, pool_size=(3, 3), strides=(2, 2), padding='same')
        with tf.variable_scope('resnet-resblock'):
            filters = 64
            for i, units in enumerate(units_per_block):
                for j in range(units):
                    strides = (1, 1)
                    if i != 0 and j == 0:
                        strides = (2, 2)
                    with tf.variable_scope('resnet-residual-unit-%d' % j):
                        inp = layer
                        res_layer = tf.layers.batch_normalization(inp)
                        res_layer = tf.nn.relu(res_layer)
                        res_layer = tf.layers.conv2d(
                            inputs=res_layer,
                            filters=filters,
                            kernel_size=(3, 3),
                            strides=strides,
                            padding='same')
                        res_layer = tf.layers.batch_normalization(res_layer)
                        res_layer = tf.nn.relu(res_layer)
                        res_layer = tf.layers.conv2d(
                            inputs=res_layer,
                            filters=filters,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')
                        shortcut = inp
                        if tf.not_equal(tf.shape(inp), tf.shape(res_layer)):
                            shortcut = tf.layers.conv2d(
                                inputs=res_layer,
                                filters=filters,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same')
                        layer = tf.add(inp, res_layer)
                layer = ResidualBlock(units, filters, residual_unit_cls, is_first_block=(i == 0))(current)
                filters *= 2
        layer = tf.nn.relu(layer)
        layer = tf.layers.avg_pooling2d(pool_size=(7, 7), strides=(1, 1))
        layer = tf.contrib.layers.flatten(layer)
        layer = tf.layers.dense(layer, units=num_classes, activation=tf.sigmoid)
        return layer


if __name__ == '__main__':
    resnet34 = get_34((224, 224, 3), 1000)
    plot_model(resnet34, to_file='resnet_34png')
    resnet101 = get_101((224, 224, 3), 1000)
    plot_model(resnet101, to_file='resnet_101.png')
