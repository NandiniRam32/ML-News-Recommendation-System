import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

news_df = pd.read_csv('data/news.tsv', sep='\t', header=None, usecols=[0, 3, 4],
                      names=['news_id', 'title', 'abstract'])

# Combine title and abstract columns
news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['abstract'].fillna('')

# Remove rows with missing or empty text
news_df = news_df[news_df['text'].notna() & (news_df['text'] != '')]

# Randomly sample a subset of the data
sample_size = 40000  # Adjust the sample size as needed
news_df = news_df.sample(n=sample_size, random_state=42)

# Load the pre-computed embeddings
embeddings_df = pd.read_pickle("news_embeddings.pkl")

# Merge the embeddings with the news_df DataFrame
news_df = news_df.merge(embeddings_df, on='news_id', how='inner')

# Combine title and abstract embeddings
news_df['embeddings'] = news_df.apply(lambda x: np.concatenate((x['title_embedding'],
                                                                x['abstract_embedding'] if isinstance(x['abstract_embedding'], np.ndarray) else np.zeros_like(x['title_embedding']))),
                                      axis=1)

# Convert the embeddings to a NumPy array
X = np.array(news_df['embeddings'].tolist())

# Perform K-means clustering for different numbers of clusters
max_clusters = 30
db_scores = []
wcss = []

for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    db_index = davies_bouldin_score(X, labels)
    db_scores.append(db_index)
    wcss.append(kmeans.inertia_)
    print("at ", k)

# Plot number of clusters vs. Davies-Bouldin index
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(2, max_clusters + 1), y=db_scores, marker='o')
plt.title("Number of Clusters vs. Davies-Bouldin Index", fontsize=16)
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("Davies-Bouldin Index", fontsize=12)
plt.xticks(range(2, max_clusters + 1), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()

# Plot the elbow curve
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(2, max_clusters + 1), y=wcss, marker='o')
plt.title("Elbow Method", fontsize=16)
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("WCSS", fontsize=12)
plt.xticks(range(2, max_clusters + 1), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()