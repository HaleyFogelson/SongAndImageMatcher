'''Reads the audio features from songs in a Spotify playlist and creates a csv'''
import pandas as pd
import spotipy
import spotipy.util as util

from spotipy.oauth2 import SpotifyClientCredentials

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

cid =""
secret = ""
username = ""

def getSP():


    client_credentials_manager = SpotifyClientCredentials(client_id=cid,
                                                        client_secret=secret)

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    scope = 'playlist-modify-private playlist-modify-public playlist-read-private user-library-read'
    token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri='https://localhost/')
    if token:
        sp = spotipy.Spotify(auth=token)
        print("Success?")

    else:
        print("Can't get token for", username)
    return sp

# Return a list of ids for all the songs in the given playlist
def getIDs(tracks, songs):
    while tracks['next']:
        tracks = sp.next(tracks)
        for item in tracks["items"]:
            songs.append(item)
    ids = []
    for i in range(len(songs)):
        ids.append(songs[i]['track']['id'])
    return ids

def addFeatures(f, playlist):
    tracks = playlist["tracks"]
    songs = tracks["items"]
    ids = getIDs(tracks, songs)
    k = 0
    for i in range(0, len(ids), 50):
        sp = getSP()
        audio_features = sp.audio_features(ids[i:i+50])
        for track in audio_features:
            if track:
                track['id'] = ids[k]
                track['song_title'] = songs[k]['track']['name']
                track['artist'] = songs[k]['track']['artists'][0]['name']
                f.append(track)
            k = k + 1

sp = getSP()

songs_playlist = sp.user_playlist('dylrobinson22', '23LcX1S7f04C8homMmZ03Z')

song_features = []

addFeatures(song_features, songs_playlist)

songs_df = pd.DataFrame(song_features)

songs_df.to_csv('songs.csv')