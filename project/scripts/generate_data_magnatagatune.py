import csv
import multiprocessing
import os
import _pickle as pickle

import click
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
CLASSES_DICT = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_SHAPE = (128, 628)
pool = None  # Global pool intialized in __main__


def get_class_vector(genres):
    class_indices = np.array([CLASSES_DICT[g] for g in genres])
    vector = np.zeros(len(CLASSES))
    vector[class_indices] = 1
    return vector


# A few spectograms have shape a bit bigger or smaller than DEFAULT_SHAPE
def fix_shape(spectogram, default_shape=DEFAULT_SHAPE):
    if spectogram.shape[1] < default_shape[1]:
        diff = default_shape[1] - spectogram.shape[1]
        return np.append(spectogram, np.zeros((spectogram.shape[0], diff)), axis=1)
    if spectogram.shape[1] > default_shape[1]:
        return spectogram[:, :default_shape[1]]
    return spectogram


def process_song(song_path, label, index, destination_dir, overwrite=False):
    basename, extension = os.path.splitext(song_path)
    if extension not in ('.mp3', '.au'):
        return
    if not os.path.isfile(song_path):
        print('NOT FOUND: ', song_path)
        return
    npfile = os.path.join(destination_dir, str(index) + '.npy')
    if not overwrite and os.path.isfile(npfile):
        return
    try:
        y, sr = librosa.load(song_path, mono=True)
    except:
        print('FAILED WITH: ', song_path)
        return None, None
    spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=1024)
    spectogram = librosa.power_to_db(spectogram, ref=np.max)
    tfrecord_filename = os.path.join(destination_dir, str(index) + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    spectogram_raw = spectogram.T.tostring()
    label_raw = np.array(label, dtype=np.uint8).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spectogram_raw])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
    }))
    writer.write(example.SerializeToString())


def get_annotations_dict(dirname):
    annotations_csv = os.path.join(dirname, 'annotations_final.csv')
    d = {}
    with open(annotations_csv, 'r') as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for i, row in enumerate(reader):
            full_path_song = os.path.join(dirname, row['mp3_path'])
            if not os.path.isfile(full_path_song):
                print('NOT FOUND: ', row['mp3_path'])
            genres = [g for g in CLASSES if row[g] == '1']
            if not genres:
                continue
            d[row['mp3_path']] = get_class_vector(genres)
    return d


def split_annotations(annotations_dict, percent_test=0.2, percent_val=0.2, percent_dev=0.2):
    X_filenames = list(annotations_dict.keys())
    y = [annotations_dict[x] for x in X_filenames]
    X_train, X_test, y_train, y_test = train_test_split(X_filenames, y, test_size=percent_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=percent_val)
    _, X_dev, _, y_dev = train_test_split(X_filenames, y, test_size=percent_dev)
    return X_train, y_train, X_test, y_test, X_val, y_val, X_dev, y_dev


def create_dir(path, dirname):
    new_dir = os.path.join(path, dirname)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def write_pickle(filename, data):
    with open(filename, 'wb') as f:
        f.write(pickle.dumps(data))
        f.close()


def save_partial_dataset(index, dirname, X, y, full_path, overwrite):
    n = len(X)
    W = multiprocessing.cpu_count()
    lo = int(index*(n / W))
    hi = int((index + 1)*(n / W))
    print('  Processing %d to %d out of %d' % (lo, hi - 1, n))
    for i, x in enumerate(X[lo:hi]):
        label = y[lo+i]
        process_song(os.path.join(dirname, x), label, i + lo, full_path, overwrite=overwrite)
    return True


def save_dataset(dirname, dataset_type, X, y, overwrite=False):
    global pool
    full_path = os.path.join(dirname, dataset_type)
    labels_file = os.path.join(dirname, dataset_type, 'labels.pickle')
    filenames_file = os.path.join(dirname, dataset_type, 'filenames.pickle')

    if not os.path.isfile(labels_file) or overwrite:
        write_pickle(labels_file, y[:20])
    if not os.path.isfile(filenames_file) or overwrite:
        write_pickle(filenames_file, X[:20])

    results = []
    print('Processing %s dataset' % (dataset_type,))
    for i in range(multiprocessing.cpu_count()):
        results.append(pool.apply_async(save_partial_dataset, args=(i, dirname, X[:20], y[:20], full_path, overwrite)))
    assert all([r.get() for r in results])


@click.command()
@click.argument('dataset_dir', nargs=1)
@click.option('--overwrite', default=False, is_flag=True)
@click.option('--percent-val', default=0.2, type=float)
@click.option('--percent-test', default=0.2, type=float)
@click.option('--percent-dev', default=0.2, type=float)
def main(dataset_dir, overwrite, percent_val, percent_test, percent_dev):
    annotations_dict = get_annotations_dict(dataset_dir)
    create_dir(dataset_dir, 'train')
    create_dir(dataset_dir, 'test')
    create_dir(dataset_dir, 'val')
    create_dir(dataset_dir, 'dev')
    X_train, y_train, X_test, y_test, X_val, y_val, X_dev, y_dev = split_annotations(
        annotations_dict, percent_test=percent_test, percent_val=percent_val, percent_dev=percent_dev)
    save_dataset(dataset_dir, 'train', X_train, y_train, overwrite=overwrite)
    save_dataset(dataset_dir, 'test', X_test, y_test, overwrite=overwrite)
    save_dataset(dataset_dir, 'val', X_val, y_val, overwrite=overwrite)
    save_dataset(dataset_dir, 'dev', X_dev, y_dev, overwrite=overwrite)


if __name__ == '__main__':
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    main()
