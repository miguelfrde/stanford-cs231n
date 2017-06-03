import os
import click

from get_preview_url import get_url


OAUTH_CLIENT_KEY='7d54pvgbj95q'
SUBSET_DIR = 'datasets/MillionSongSubset/data'
DATASET_DIR = 'datasets/MillionSong/data'


@click.command()
@click.option('--real-dataset', default=False)
def main(real_dataset):
    path = 'datasets/MillionSongSubset/data' if not real_dataset else 'datasets/MillionSong/data'
    path = DATASET_DIR if real_dataset else SUBSET_DIR
    fetch_songs_in_dir(path)


def fetch_songs_in_dir(dirname):
    for filename in os.listdir(dirname):
        song_path = os.path.join(dirname, filename)
        if os.path.isdir(song_path):
            fetch_songs_in_dir(song_path)
            continue
        basename, extension = os.path.splitext(song_path)
        if extension != '.h5':
            return
        get_url(song_path, OAUTH_CLIENT_KEY)


if __name__ == '__main__':
    os.environ['DIGITAL7_API_KEY'] = OAUTH_CLIENT_KEY
    main()
