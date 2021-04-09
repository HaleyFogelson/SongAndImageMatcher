import pandas as pd
import sqlite3
import spotipy

from spotipy.oauth2 import SpotifyClientCredentials


songs_db_con = sqlite3.connect('track_metadata.db')

songs_df = pd.read_sql_query("SELECT * FROM songs", songs_db_con)

print(songs_df.head())

songs_db_con.close()

song_titles = songs_df['title']

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

song_ids = []

for title in song_titles:
    query = title.replace(' ', '+')
    search_result = spotify.search(query)
    if len(search_result['tracks']['items']) > 0:
        song = search_result['tracks']['items'][0]
        song_ids.append(song['id'])
        print(song['name'])

print(song_ids)