import csv
import os
import _pickle as pickle

import click
import librosa
import numpy as np
import tensorflow as tf


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


def process_song(song_path, genres, overwrite=False):
    basename, extension = os.path.splitext(song_path)
    if extension not in ('.mp3', '.au'):
        return None, None
    if not os.path.isfile(song_path):
        print('NOT FOUND: ', song_path)
        return None, None
    picklefile = basename + '.pickle'
    if not overwrite and os.path.isfile(picklefile):
        with open(picklefile, 'rb') as f:
            spectogram, class_vector = pickle.load(f)
            return fix_shape(spectogram), class_vector
    try:
        y, sr = librosa.load(song_path, mono=True)
    except:
        print('FAILED WITH: ', song_path)
        return None, None
    spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=1024)
    spectogram = librosa.power_to_db(spectogram, ref=np.max)
    class_vector = get_class_vector(genres)
    with open(basename + '.pickle', 'wb') as f:
        f.write(pickle.dumps((spectogram, class_vector)))
        f.close()
    return fix_shape(spectogram).T, class_vector


def process_songs(dirname, overwrite=False):
    annotations_csv = os.path.join(dirname, 'annotations_final.csv')
    genre_vecs = []
    spectograms = []
    with open(annotations_csv, 'r') as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for i, row in enumerate(reader):
            genres = [g for g in CLASSES if row[g] == '1']
            if not genres:
                continue
            full_path_song = os.path.join(dirname, row['mp3_path'])
            spectogram, genre_vec = process_song(full_path_song, genres, overwrite=overwrite)
            if spectogram is None:
                continue
            spectograms.append(spectogram)
            genre_vecs.append(genre_vec)
    return spectograms, genre_vecs


@click.command()
@click.argument('dataset_dir', nargs=1)
@click.option('--overwrite', default=False, is_flag=True)
def main(dataset_dir, overwrite):
    spectograms, classes = process_songs(dataset_dir, overwrite=overwrite)
    print(len(spectograms), spectograms[0].shape)
    print(len(classes), classes[0].shape)
    X = np.vstack(spectograms).reshape((len(spectograms),) + DEFAULT_SHAPE)
    y = np.vstack(classes).reshape((len(spectograms), len(CLASSES)))
    print(X.shape)
    print(y.shape)
    assert np.array_equal(X[0], spectograms[0])
    X = X.transpose(0, 2, 1)
    with tf.Graph().as_default():
        X_init = tf.placeholder(tf.float32, shape=X.shape)
        y_init = tf.placeholder(tf.float32, shape=y.shape)
        X_data = tf.Variable(X_init, name='X_data')
        y_data = tf.Variable(y_init, name='y_data')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={X_init: X, y_init: y})
            saver.save(sess, os.path.join(dataset_dir, 'data.ckpt'))

if __name__ == '__main__':
    main()
