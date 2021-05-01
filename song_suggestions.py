import pandas as pd
import numpy as np

images_df = pd.read_csv('clustered_images.csv')
songs_df = pd.read_csv('clustered_songs.csv')


image_cluster_song_cluster_map = {
    0: 3, #nothing
    1: 6, #happy
    2: 4, #surprised
    3: 1, #very happy
    4: 2, #fearful
    5: 0, # neutral 0
    6: 5 # also happy
}

def suggest_song(row):
    image_cluster = row['cluster'].values[0]
    song_cluster = image_cluster_song_cluster_map[int(image_cluster)]

    is_cluster = songs_df['cluster'] == song_cluster

    songs_in_cluster = songs_df[is_cluster]

    song = songs_in_cluster.sample()

    song_title = song['song_title'].values[0]
    song_artist = song['artist'].values[0]
    song_uri = song['uri'].values[0]

    print()
    print('For image ' + row['file name'].values[0] + ' the suggested song is: ')
    print(song_title + ' by ' + song_artist)
    print('Listen on Spotify by searching for the uri  ' + song_uri)
    print()

suggest_song(images_df.sample())
suggest_song(images_df.sample())
suggest_song(images_df.sample())
suggest_song(images_df.sample())
suggest_song(images_df.sample())
suggest_song(images_df.sample())
suggest_song(images_df.sample())
suggest_song(images_df.sample())