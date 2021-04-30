'''Run clustering algorithms on our song dataset'''
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

song_df = pd.read_csv('clustered_songs.csv')
print(song_df.head())

features = ["valence",
            "energy"]

n_clusters = 7

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(song_df[features])
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
spectral.fit(song_df[features])


# print('K-means cluster centers: ')
# print(kmeans.cluster_centers_)

# print('K-means labels:')
# print(kmeans.labels_)



cluster_groups = kmeans.labels_
cluster_groups_2 = spectral.labels_
# for i in range(len(song_features)):
#     song_features[i]['cluster'] = cluster_groups[i]



# songs_clustered_df = pd.DataFrame(song_features)

songs_clustered_df = song_df
songs_clustered_df.drop('cluster', axis=1, inplace=True)
songs_clustered_df['cluster'] = cluster_groups
songs_clustered_df['cluster2'] = cluster_groups_2

songs_clustered_df.to_csv('clustered_songs2.csv', index=False)

print(songs_clustered_df.head())

sns.set_theme()

for feature in features:
    sns.histplot(data=songs_clustered_df, x=feature, hue='cluster', bins=30, multiple='stack', palette=sns.color_palette("husl", n_clusters))
    plt.show()
    sns.histplot(data=songs_clustered_df, x=feature, hue='cluster2', bins=30, multiple='stack', palette=sns.color_palette("husl", n_clusters))
    plt.show()

sns.scatterplot(data=songs_clustered_df, x='valence', y='energy', hue='cluster', palette=sns.color_palette("husl", n_clusters))
plt.show()
sns.scatterplot(data=songs_clustered_df, x='valence', y='energy', hue='cluster2', palette=sns.color_palette("husl", n_clusters))
plt.show()