import os
import re
import _pickle as pickle

import click
import librosa
import numpy as np


CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
CLASSES_DICT = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_SHAPE = (128, 647)


def get_genre_from_path(song_path):
    m = re.search(r'([a-z]+).\d{5}.(au|mp3)', song_path)
    return m.groups(0)[0]


def get_class_vector(class_index):
    vector = np.zeros(len(CLASSES))
    vector[class_index] = 1
    return vector


# A few spectograms have shape a bit bigger or smaller than DEFAULT_SHAPE
def fix_shape(spectogram, default_shape=DEFAULT_SHAPE):
    if spectogram.shape[1] < default_shape[1]:
        diff = default_shape[1] - spectogram.shape[1]
        return np.append(spectogram, np.zeros((spectogram.shape[0], diff)), axis=1)
    if spectogram.shape[1] > default_shape[1]:
        return spectogram[:, :default_shape[1]]
    return spectogram


def process_song(song_path, overwrite=False):
    basename, extension = os.path.splitext(song_path)
    if extension not in ('.mp3', '.au'):
        return None, None
    picklefile = basename + '.pickle'
    if not overwrite and os.path.isfile(picklefile):
        with open(picklefile, 'rb') as f:
            spectogram, class_index = pickle.load(f)
            return fix_shape(spectogram), get_class_vector(class_index)
    y, sr = librosa.load(song_path, mono=True)
    spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=1024)
    spectogram = librosa.power_to_db(spectogram, ref=np.max)
    class_name = get_genre_from_path(song_path)
    with open(basename + '.pickle', 'wb') as f:
        f.write(pickle.dumps((spectogram, CLASSES_DICT[class_name])))
        f.close()
    return fix_shape(spectogram).T, get_class_vector(CLASSES_DICT[class_name])


def process_songs_in_dir(dirname, overwrite=False):
    spectograms = []
    classes = []
    for filename in os.listdir(dirname):
        song_path = os.path.join(dirname, filename)
        if os.path.isdir(song_path):
            spcs, clss = process_songs_in_dir(song_path, overwrite=overwrite)
            spectograms.extend(spcs)
            classes.extend(clss)
            continue
        song_spectogram, class_vector = process_song(song_path, overwrite=overwrite)
        if song_spectogram is not None:
            spectograms.append(song_spectogram)
            classes.append(class_vector)
    return spectograms, classes


@click.command()
@click.argument('dataset_dir', nargs=1)
@click.option('--overwrite', default=False, is_flag=True)
def main(dataset_dir, overwrite):
    spectograms, classes = np.array(process_songs_in_dir(dataset_dir, overwrite=overwrite))
    X = np.vstack(spectograms).reshape((len(spectograms),) + DEFAULT_SHAPE)
    y = np.vstack(classes).reshape((len(spectograms), len(CLASSES)))
    assert np.array_equal(X[0], spectograms[0])
    with open(os.path.join(dataset_dir, 'data.pickle'), 'wb') as f:
        f.write(pickle.dumps([X, y]))


if __name__ == '__main__':
    main()
