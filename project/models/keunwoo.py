import tensorflow as tf


def keunwoo(x, num_classes, batchnorm=True):
    conv_filters = (128, 384, 768, 2048)
    pool_sizes = ((2,4), (4,5), (3,8), (4,8))
    with tf.variable_scope('keunwoo'):
        layer = x
        for i, conv_filter, pool_size in enumerate(zip(conv_filters, pool_sizes)):
            with tf.variable_scope('conv%d' % i):
                layer = tf.layers.conv2d(
                    inputs=layer,
                    filters=conv_filter,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu)
                if batchnorm:
                    layer = tf.layers.batch_normalization(layer)
                layer = tf.layers.max_pooling2d(layer, pool_size=pool_size, strides=pool_size)
		layer = tf.layers.dense(layer, units=num_classes, activation=tf.sigmoid)
		return layer
