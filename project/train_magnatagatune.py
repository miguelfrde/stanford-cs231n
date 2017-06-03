import multiprocessing
import os
import re
import _pickle as pickle

import tensorflow as tf
import tensorflow.contrib.slim.nets
import numpy as np

from models import spotify


PATH_MAGNATAGATUNE = 'datasets/magnatagatune'

INPUT_SHAPE = (628, 128)
CLASSES = [
    'classical', 'instrumental', 'electronica', 'techno',
    'male voice', 'rock', 'ambient', 'female voice', 'opera',
    'indian', 'choir', 'pop', 'heavy metal', 'jazz', 'new age',
    'dance', 'country', 'eastern', 'baroque', 'funk', 'hard rock',
    'trance', 'folk', 'oriental', 'medieval', 'irish', 'blues',
    'middle eastern', 'punk', 'celtic', 'arabic', 'rap',
    'industrial', 'world', 'hip hop', 'disco', 'soft rock',
    'jungle', 'reggae', 'happy',
]
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = multiprocessing.cpu_count()


def _load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def _list_dataset(dataset_name):
    dataset_path = os.path.join(PATH_MAGNATAGATUNE, dataset_name)
    labels_file = os.path.join(dataset_path, 'labels.pickle')
    filenames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                 if re.match(r'\d+\.tfrecord', f)]
    print(filenames, dataset_path)
    labels = _load_pickle(labels_file)
    return filenames, np.asarray(labels)


def _parse_function(example_proto):
    features = {
        'X': tf.FixedLenFeature((), tf.string),
        'y': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    spectogram = tf.decode_raw(parsed_features['X'], tf.float64)
    spectogram = tf.cast(spectogram, tf.float32)
    spectogram = tf.reshape(spectogram, INPUT_SHAPE)
    label = tf.decode_raw(parsed_features['y'], tf.uint8)
    label = tf.reshape(label, [len(CLASSES)])
    return spectogram, label


def _init_datasets(train_filenames, val_filenames):
    train_filenames = tf.constant(train_filenames)
    train_dataset = tf.contrib.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    batched_train_dataset = train_dataset.batch(BATCH_SIZE)

    val_filenames = tf.constant(val_filenames)
    val_dataset = tf.contrib.data.TFRecordDataset(val_filenames)
    val_dataset = val_dataset.map(_parse_function)
    val_dataset = val_dataset.shuffle(buffer_size=10000)
    batched_val_dataset = val_dataset.batch(BATCH_SIZE)

    iterator = tf.contrib.data.Iterator.from_structure(
        batched_train_dataset.output_types, batched_train_dataset.output_shapes)
    spectograms, labels = iterator.get_next()

    train_init_op = iterator.make_initializer(batched_train_dataset)
    val_init_op = iterator.make_initializer(batched_val_dataset)

    return spectograms, labels, train_init_op, val_init_op


def train(initial_learning_rate, learning_rate_decay=0.96):
    train_filenames, train_labels = _list_dataset('train')
    val_filenames, val_labels = _list_dataset('val')

    graph = tf.Graph()
    with graph.as_default():
        spectograms, labels, train_init_op, val_init_op = _init_datasets(
            train_filenames, val_filenames)

        is_training = tf.placeholder(tf.bool)
        output_layer = spotify.get_tf(spectograms, len(CLASSES), activation='sigmoid')
        model_variables = tf.contrib.framework.get_variables('spotify_tf')
        model_init = tf.variables_initializer(model_variables)

        tf.losses.mean_squared_error(labels=labels, predictions=output_layer)
        loss = tf.losses.get_total_loss()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, 100000, learning_rate_decay, staircase=True)
        optimizer =  tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(lr=learning_rate)
        train_op = optimizer.minimize(loss)

        correct_prediction = tf.equal(
            tf.round(tf.nn.sigmoid(output_layer)),
            tf.round(tf.cast(labels, tf.float32)))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.get_default_graph().finalize()

    with tf.Session(graph=graph) as sess:
        sess.run(model_init)
        for epoch in range(EPOCHS):
            print('Epoch %d / %d' % (epoch + 1, EPOCHS))
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('  Train accuracy: %f' % train_acc)
            print('  Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    main()
