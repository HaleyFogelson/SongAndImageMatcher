import pandas as pd
import spotipy
import spotipy.util as util

from spotipy.oauth2 import SpotifyClientCredentials

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

cid ="38a975768416491cbf5ac70921868d46"
secret = ""
username = "dylrobinson22"

def getSP():


    client_credentials_manager = SpotifyClientCredentials(client_id=cid,
                                                        client_secret=secret)

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    scope = 'playlist-modify-private playlist-modify-public playlist-read-private user-library-read'
    token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri='http://localhost/')
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

song_df = pd.DataFrame(song_features)
print(song_df.head())

features = ["danceability", "loudness", "valence",
            "energy", "instrumentalness", "acousticness",
            "key", "speechiness", "mode", "time_signature"]

kmeans = KMeans(n_clusters=8)
kmeans.fit(song_df[features])

print('K-means cluster centers: ')
print(kmeans.cluster_centers_)

print('K-means labels:')
print(kmeans.labels_)

cluster_groups = kmeans.labels_

for i in range(len(song_features)):
    song_features[i]['cluster'] = cluster_groups[i]

songs_clustered_df = pd.DataFrame(song_features)

# TODO
# tempo = songs_clustered_df['tempo', 'cluster']
# dance = songs_clustered_df['danceability', 'cluster']
# duration = songs_clustered_df['duration_ms', 'cluster']
# loudness = songs_clustered_df['loudness', 'cluster']
# speechiness = songs_clustered_df['speechiness', 'cluster']
# valence = songs_clustered_df['valence', 'cluster']
# energy = songs_clustered_df['energy', 'cluster']
# acousticness = songs_clustered_df['acousticness', 'cluster']
# key = songs_clustered_df['key', 'cluster']
# instrumentalness = songs_clustered_df['instrumentalness', 'cluster']

# # Tempo Graph
# fig = plt.figure(figsize=(12,8))
# plt.title("Song Tempo Like / Dislike Distribution")
# pos_tempo.hist(alpha=0.7, bins=30, label='positive')
# neg_tempo.hist(alpha=0.7, bins=30, label='negative')
# plt.legend(loc='upper right')
# plt.show()

# fig2 = plt.figure(figsize=(15,15))

# #Danceability
# ax3 = fig2.add_subplot(331)
# ax3.set_xlabel('Danceability')
# ax3.set_ylabel('Count')
# ax3.set_title('Song Danceability Like Distribution')
# pos_dance.hist(alpha= 0.5, bins=30)
# ax4 = fig2.add_subplot(331)
# neg_dance.hist(alpha= 0.5, bins=30)

# #Duration_ms
# ax5 = fig2.add_subplot(332)
# ax5.set_xlabel('Duration')
# ax5.set_ylabel('Count')
# ax5.set_title('Song Duration Like Distribution')
# pos_duration.hist(alpha= 0.5, bins=30)
# ax6 = fig2.add_subplot(332)
# neg_duration.hist(alpha= 0.5, bins=30)

# #Loudness
# ax7 = fig2.add_subplot(333)
# ax7.set_xlabel('Loudness')
# ax7.set_ylabel('Count')
# ax7.set_title('Song Loudness Like Distribution')
# pos_loudness.hist(alpha= 0.5, bins=30)
# ax8 = fig2.add_subplot(333)
# neg_loudness.hist(alpha= 0.5, bins=30)

# #Speechiness
# ax9 = fig2.add_subplot(334)
# ax9.set_xlabel('Speechiness')
# ax9.set_ylabel('Count')
# ax9.set_title('Song Speechiness Like Distribution')
# pos_speechiness.hist(alpha= 0.5, bins=30)
# ax10 = fig2.add_subplot(334)
# neg_speechiness.hist(alpha= 0.5, bins=30)

# #Valence
# ax11 = fig2.add_subplot(335)
# ax11.set_xlabel('Valence')
# ax11.set_ylabel('Count')
# ax11.set_title('Song Valence Like Distribution')
# pos_valence.hist(alpha= 0.5, bins=30)
# ax12 = fig2.add_subplot(335)
# neg_valence.hist(alpha= 0.5, bins=30)

# #Energy
# ax13 = fig2.add_subplot(336)
# ax13.set_xlabel('Energy')
# ax13.set_ylabel('Count')
# ax13.set_title('Song Energy Like Distribution')
# pos_energy.hist(alpha= 0.5, bins=30)
# ax14 = fig2.add_subplot(336)
# neg_energy.hist(alpha= 0.5, bins=30)

# #Key
# ax15 = fig2.add_subplot(337)
# ax15.set_xlabel('Key')
# ax15.set_ylabel('Count')
# ax15.set_title('Song Key Like Distribution')
# key.hist(alpha= 0.5, bins=30)


# plt.show()