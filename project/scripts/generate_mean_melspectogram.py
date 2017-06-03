import os
import re

import click
import numpy as np
import tensorflow as tf


SHAPE = (628, 128)


def save_tfrecord(mean, stddev, tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    stddev_raw = stddev.tostring()
    mean_raw = mean.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'stddev': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stddev_raw])),
        'mean': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mean_raw])),
    }))
    writer.write(example.SerializeToString())


@click.command()
@click.argument('dataset_dir', nargs=1)
def main(dataset_dir):
    sum1 = np.zeros(SHAPE)
    sum2 = np.zeros(SHAPE)
    n = 0
    sess = tf.InteractiveSession()

    for i, filename in enumerate(os.listdir(dataset_dir)):
        if i % 1000 == 0:
            print('Processing #%d...' % i)
        basename, extension = os.path.splitext(filename)
        if not re.match(r'\d+\.tfrecord', filename):
            continue
        record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(dataset_dir, filename))
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            x = example.features.feature['X'].bytes_list.value[0]
            x = np.fromstring(x, dtype=np.float64)
            x = x.reshape(SHAPE)
            n += 1
            sum1 += x
            sum2 += x*x
    mean = sum1 / n
    variance = (n*sum2 - sum1*sum1) / n**2
    stddev = np.sqrt(variance)
    save_tfrecord(mean, stddev, os.path.join(dataset_dir, 'stats.tfrecord'))


if __name__ == '__main__':
    main()
