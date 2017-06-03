import json
import oauth2 as oauth


OAUTH_CLIENT_KEY = '7d54pvgbj95q'
OAUTH_CLIENT_SECRET = 'tprh5qpdy8ec3ctv'
OAUTH_REQUEST_TOKEN = 'Y25GGN7'
OAUTH_REQUEST_SECRET = 'UFdYm3vtsEaYvAkeEIfW5A=='
OAUTH_ACCESS_TOKEN = 'iapMJgwIl0uIiLuWS7yleA'
OAUTH_ACCESS_SECRET = 'ebzeSfQGTEu38VhucA0C3A'
SEVEN_DIGITAL_REQUEST_TOKEN_URL = 'https://api.7digital.com/1.2/oauth/requesttoken'
AUTHORIZATION_URL = 'https://account.7digital.com/{oauth_key}/oauth/authorise?oauth_token={oauth_token}'
ACCESS_TOKEN_URL = 'https://api.7digital.com/1.2/oauth/accesstoken'


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
    print('Authorization url:', AUTHORIZATION_URL.format(oauth_key=OAUTH_CLIENT_KEY, oauth_token=oauth_token))
    print(content)
    print(oauth_token, oauth_token_secret)
    return oauth_token, oauth_token_secret


def request_access_token():
    client = oauth.Client(
        oauth.Consumer(OAUTH_CLIENT_KEY, OAUTH_CLIENT_SECRET),
        token=oauth.Token(OAUTH_REQUEST_TOKEN, OAUTH_REQUEST_SECRET))
    response, content = client.request(
        ACCESS_TOKEN_URL,
        headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
        body='country=ww',
        method='POST')
    tokens_json = json.loads(content)['oauth_request_token']
    oauth_token, oauth_token_secret = tokens_json['oauth_token'], tokens_json['oauth_token_secret']
    return oauth_token, oauth_token_secret


#request_token()
#input()
#request_access_token()
