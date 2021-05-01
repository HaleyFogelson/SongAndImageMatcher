from emotions import commandLineAlgorthmsMain
import pandas as pd

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

def predict(new_image_df):
    image_df = pd.read_csv('result.csv')

    def happy_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'happy':
                x = x + 1
        return x

    def angry_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'angry':
                x = x + 1
        return x

    def sad_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'sad':
                x = x + 1
        return x

    def neutral_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'neutral':
                x = x + 1
        return x

    def fearful_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'fearful':
                x = x + 1
        return x

    def surprised_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'surprised':
                x = x + 1
        return x

    def disgusted_counter(row):
        emotions = row['emotions']
        emotions = emotions.lower().replace(',', '').split()
        x = 0
        for emotion in emotions:
            if emotion == 'disgusted':
                x = x + 1
        return x

    image_df['happy'] = image_df.apply(lambda row: happy_counter(row), axis=1)
    image_df['angry'] = image_df.apply(lambda row: angry_counter(row), axis=1)
    image_df['sad'] = image_df.apply(lambda row: sad_counter(row), axis=1)
    image_df['neutral'] = image_df.apply(lambda row: neutral_counter(row), axis=1)
    image_df['fearful'] = image_df.apply(lambda row: fearful_counter(row), axis=1)
    image_df['surprised'] = image_df.apply(lambda row: surprised_counter(row), axis=1)
    image_df['disgusted'] = image_df.apply(lambda row: disgusted_counter(row), axis=1)

    features = ['happy', 'angry', 'sad', 'neutral', 'fearful', 'surprised', 'disgusted']

    n_clusters = 7

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(image_df[features])

    cluster_groups = kmeans.labels_

    image_df['cluster'] = cluster_groups

    predicted_cluster = kmeans.predict(new_image_df[features])[0]

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

    song_cluster = image_cluster_song_cluster_map[predicted_cluster]

    is_cluster = songs_df['cluster'] == song_cluster

    songs_in_cluster = songs_df[is_cluster]

    song = songs_in_cluster.sample()

    song_title = song['song_title'].values[0]
    song_artist = song['artist'].values[0]
    song_uri = song['uri'].values[0]

    print()
    print('The suggested song is: ')
    print(song_title + ' by ' + song_artist)
    print('Listen on Spotify by searching for the uri  ' + song_uri)
    print()



if __name__ == "__main__":
    emotion_list = commandLineAlgorthmsMain()
    print()
    print('These emotions were found in the image: ')
    print(emotion_list)

    emotions_dict = {'happy': [0], 'sad': [0], 'angry': [0], 'neutral': [0], 'disgusted': [0], 'surprised': [0], 'fearful': [0]}

    for emotion in emotion_list:
        emotions_dict[emotion.lower()][0] += 1

    new_image_df = pd.DataFrame(emotions_dict)

    predict(new_image_df)    

    