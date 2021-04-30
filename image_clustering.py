import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


image_df = pd.read_csv('LiveEmotions/result.csv')

print(image_df.head())

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

image_df.to_csv('clustered_images.csv', index=False)