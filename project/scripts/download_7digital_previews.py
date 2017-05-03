import os
import json
import click

import hdf5_getters
import requests
import oauth2 as oauth


SUBSET_DIR = 'datasets/MillionSongSubset/data'
DATASET_DIR = 'datasets/MillionSong/data'

SEVEN_DIGITAL_API_COUNTRY = 'US'
SEVEN_DIGITAL_CLIP_URL = 'http://previews.7digital.com/clip/{track_id}'
SEVEN_DIGITAL_REQUEST_TOKEN_URL = 'https://api.7digital.com/1.2/oauth/requesttoken'
SEVEN_DIGITAL_REQUEST_TOKEN_URL = 'https://account.7digital.com/{oauth_key}/oauth/authorise?oauth_token={oauth_token}'
SEVEN_DIGITAL_ACCESS_TOKEN_URL = 'https://api.7digital.com/1.2/oauth/accesstoken'

OAUTH_CLIENT_KEY = os.environ['OAUTH_CLIENT_KEY']
OAUTH_CLIENT_SECRET = os.environ['OAUTH_CLIENT_KEY']
OAUTH_REQUEST_TOKEN = os.environ['OAUTH_REQUEST_TOKEN']
OAUTH_REQUEST_SECRET = os.environ['OAUTH_REQUEST_SECRET']
OAUTH_ACCESS_TOKEN = os.environ['OAUTH_ACCESS_TOKEN']
OAUTH_ACCESS_SECRET = os.environ['OAUTH_ACCESS_SECRET']


def request_token():
    consumer = oauth.Consumer(OAUTH_CLIENT_KEY, OAUTH_CLIENT_SECRET)
    client = oauth.Client(consumer)
    response, content = client.request(
        SEVEN_DIGITAL_REQUEST_TOKEN_URL,
        headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
        body='country=ww',
        method='POST'
    )
    tokens_json = json.loads(content)['oauth_request_token']
    oauth_token, oauth_token_secret = tokens_json['oauth_token'], tokens_json['oauth_token_secret']
    print('Authorization url:', SEVEN_DIGITAL_REQUEST_TOKEN_URL.format(
        oauth_key=OAUTH_CLIENT_KEY, oauth_token=oauth_token))
    print(oauth_token, oauth_token_secret)
    return oauth_token, oauth_token_secret


def request_access_token():
    client = oauth.Client(
        oauth.Consumer(OAUTH_CLIENT_KEY, OAUTH_CLIENT_SECRET),
        token=oauth.Token(OAUTH_REQUEST_TOKEN, OAUTH_REQUEST_SECRET))
    response, content = client.request(
        SEVEN_DIGITAL_ACCESS_TOKEN_URL,
        headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
        body='country=ww',
        method='POST')
    tokens_json = json.loads(content)['oauth_request_token']
    oauth_token, oauth_token_secret = tokens_json['oauth_token'], tokens_json['oauth_token_secret']
    print(oauth_token, oauth_token_secret)
    return oauth_token, oauth_token_secret


def get_clip_url(track_id):
    return SEVEN_DIGITAL_CLIP_URL.format(
        track_id=track_id,
        oauth_key=OAUTH_ACCESS_TOKEN,
        country='ww')


def fetch_song_from_h5(h5_filepath):
    basename, extension = os.path.splitext(h5_filepath)
    if extension != '.h5':
        return
    audio_filepath = basename + '.mp3'
    h5 = hdf5_getters.open_h5_file_read(h5_filepath)
    track_id = hdf5_getters.get_track_7digitalid(h5)
    track_name = hdf5_getters.get_title(h5)
    artist_name = hdf5_getters.get_artist_name(h5)
    h5.close()

    consumer = oauth.Consumer(OAUTH_CLIENT_KEY, OAUTH_CLIENT_SECRET)
    token = oauth.Token(OAUTH_ACCESS_TOKEN, OAUTH_ACCESS_SECRET)
    request = oauth.Request.from_consumer_and_token(
        consumer,
        http_url=get_clip_url(track_id),
        is_form_encoded=True,
        parameters={'country': 'ww'})
    signing_method = oauth.SignatureMethod_HMAC_SHA1()
    request.sign_request(signing_method, consumer, token)
    url = request.to_url()
    r = requests.get(url)
    if r.status_code not in (requests.codes.ok, requests.codes.not_found):
        print(r.status_code, r.headers, r.content)
        exit()
    if r.status_code == requests.codes.ok:
        print('FETCHED track {0} {1} {2}'.format(
            track_id, artist_name, track_name))
        with open(audio_filepath, 'wb') as f:
            f.write(r.content)
    else:
        print('FAILED TO FETCH track {0} {1} {2}'.format(
            track_id, artist_name, track_name))


def fetch_songs_in_dir(dirname):
    for filename in os.listdir(dirname):
        song_path = os.path.join(dirname, filename)
        if os.path.isdir(song_path):
            fetch_songs_in_dir(song_path)
            continue
        fetch_song_from_h5(song_path)


@click.command()
@click.option('--real-dataset', default=False)
def main(real_dataset):
    path = 'datasets/MillionSongSubset/data' if not real_dataset else 'datasets/MillionSong/data'
    path = DATASET_DIR if real_dataset else SUBSET_DIR
    fetch_songs_in_dir(path)


if __name__ == '__main__':
    main()
