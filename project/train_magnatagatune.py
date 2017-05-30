import os
import re
import _pickle as pickle

import tensorflow as tf
import numpy as np

from models import spotify


PATH_MAGNATAGATUNE = 'datasets/magnatagatune'
X_DATA_SHAPE = (15659, 628, 128)
Y_DATA_SHAPE = (15659, 40)
PERCENT_TEST = 0.2
PERCENT_VAL = 0.2
N_TRAIN = 10021
N_TEST = 3132
N_VAL = 2506
MAGNATAGATUNE_GENRES = [
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
NUM_WORKERS = 4


def _load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def _list_dataset(dataset_name):
    dataset_path = os.path.join(PATH_MAGNATAGATUNE, dataset_name)
    labels_file = os.path.join(dataset_path, 'labels.pickle')
    filenames = [os.path.join(dataset_path, '%d.pickle') % i for i, f in enumerate(os.listdir(dataset_path))
                 if re.match(r'\d+\.pickle', f)]
    labels = _load_pickle(labels_file)
    return filenames, np.asarray(labels)


def _parse_function(filename, label):
    spectogram = tf.cast(_load_pickle(filename), tf.float32)
    return spectogram, label


def _create_dataset(filenames, labels, dataset_name):
    with tf.variable_scope(dataset_name):
        dataset_filenames = tf.constant(filenames)
        dataset_labels = tf.constant(labels)
        dataset = tf.contrib.data.Dataset.from_tensor_slices((dataset_filenames, dataset_labels))
        dataset = dataset.map(_parse_function,
            num_threads=NUM_WORKERS, output_buffer_size=BATCH_SIZE)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(BATCH_SIZE)
        return batched_dataset


def _init_datasets(train_filenames, val_filenames, train_labels, val_labels):
   val_dataset = _create_dataset(val_filenames, val_labels, dataset_name='val_dataset')
   train_dataset = _create_dataset(train_filenames, train_labels, dataset_name='train_dataset')

   iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                      batched_train_dataset.output_shapes)
   spectograms, labels = iterator.get_next()

   train_init_op = iterator.make_initializer(batched_train_dataset)
   val_init_op = iterator.make_initializer(batched_val_dataset)

   return spectograms, labels, train_init_op, val_init_op


def main():
    train_filenames, train_labels = _list_dataset('train')
    val_filenames, val_labels = _list_dataset('val')

    graph = tf.Graph()
    with graph.as_default():
        spectograms, labels, train_init_op, val_init_op = _init_datasets(
            train_filenames, val_filenames, train_labels, val_labels)
        print(spectograms)

        #is_training = tf.placeholder(tf.bool)
        #output_layer = spotify.get(INPUT_SHAPE, NUM_CLASSES, activation='sigmoid')

        #tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer)
        #loss = tf.losses.get_total_loss()

        #optimizer =  tf.train.GradientDescentOptimizer(LEARNING_RATE)
        #train_op = optimizer.minimize(loss)

        ## Evaluation metrics (TODO - adjust for multiple classes)
        #prediction = tf.to_int32(tf.argmax(logits, 1))
        #correct_prediction = tf.equal(prediction, labels)
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.get_default_graph().finalize()

        #with tf.Session(graph=graph) as sess:
        #    for epoch in range(EPOCHS):
        #        print('Epoch %d / %d' % (epoch + 1, EPOCHS))
        #        sess.run(train_init_op)
        #        while True:
        #            try:
        #                _ = sess.run(train_op, {is_training: True})
        #            except tf.errors.OutOfRangeError:
        #                break

        #        # Check accuracy on the train and val sets every epoch
        #        train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
        #        val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
        #        print('  Train accuracy: %f' % train_acc)
        #        print('  Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    main()
