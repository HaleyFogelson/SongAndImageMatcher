import pandas as pd
import sqlite3
import spotipy
import spotipy.util as util

from spotipy.oauth2 import SpotifyClientCredentials


cid =""
secret = ""
username = ""

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


songs_db_con = sqlite3.connect('track_metadata.db')

songs_df = pd.read_sql_query("SELECT * FROM songs", songs_db_con)

print(songs_df.head())

songs_db_con.close()

song_titles = songs_df['title']

sp = getSP()

song_ids = []

for title in song_titles:
    query = title.replace(' ', '+')
    try:
        search_result = sp.search(query)
        if len(search_result['tracks']['items']) > 0:
            song = search_result['tracks']['items'][0]
            song_ids.append(song['id'])
            print(song['name'])
            print(len(song_ids))
    except:
        sp = getSP()
    if len(song_ids) >= 9000:
        break

for i in range(0, len(song_ids), 100):
    sp.user_playlist_add_tracks(username, '23LcX1S7f04C8homMmZ03Z', song_ids[i:i+100])

print(song_ids)