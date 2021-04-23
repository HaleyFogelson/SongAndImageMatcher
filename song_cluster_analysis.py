import pandas as pd
import numpy as np


features = ["danceability", "loudness", "valence",
            "energy", "key"]

songs_df = pd.read_csv('clustered_songs.csv')

print(songs_df.head())

for cluster in range(7):
    print('Cluster ' + str(cluster) + ':\n')
    cluster_means = songs_df[songs_df['cluster'] == cluster].mean(axis=0)
    print(cluster_means)