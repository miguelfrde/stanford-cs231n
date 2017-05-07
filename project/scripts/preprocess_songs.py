import os
import sys
import pickle

import librosa
import numpy as np


def process_song(song_path):
    basename, extension = os.path.splitext(song_path)
    if not extension in ('.mp3', '.au'):
        return
    y, sr = librosa.load(song_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=np.max), sr=sr, n_mfcc=13)
    with open(basename + '.pickle', 'wb') as f:
        f.write(pickle.dumps(mfcc))
        f.close()


def process_songs_in_dir(dirname):
    for filename in os.listdir(dirname):
        song_path = os.path.join(dirname, filename)
        if os.path.isdir(song_path):
            process_songs_in_dir(song_path)
            continue
        process_song(song_path)


if __name__ == '__main__':
    process_songs_in_dir(sys.argv[1])
