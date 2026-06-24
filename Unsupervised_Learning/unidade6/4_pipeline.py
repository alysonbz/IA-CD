import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from src.utils import load_fish_dataset

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values


scaler = StandardScaler()

kmeans = KMeans(n_clusters=4)

pipeline = make_pipeline(scaler, kmeans)

pipeline.fit(samples)

labels = pipeline.predict(samples)

df = pd.DataFrame({'labels': labels, 'species': species})

ct = pd.crosstab(df['labels'], df['species'])

print(ct)