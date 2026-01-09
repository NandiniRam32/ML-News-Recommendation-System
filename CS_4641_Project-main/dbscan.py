import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time

news_df = pd.read_csv('data/news.tsv', sep='\t', header=None, usecols=[0, 3, 4],
                      names=['news_id', 'title', 'abstract'])

# Combine title and abstract columns
news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['abstract'].fillna('')

# Remove rows with missing or empty text
news_df = news_df[news_df['text'].notna() & (news_df['text'] != '')]

# Randomly sample a subset of the data
sample_size = 10000  # Adjust the sample size as needed
news_df = news_df.sample(n=sample_size, random_state=42)

print("Data loaded and preprocessed.")

# Load the pre-computed embeddings
embeddings_df = pd.read_pickle("news_embeddings.pkl")

# Merge the embeddings with the news_df DataFrame
news_df = news_df.merge(embeddings_df, on='news_id', how='inner')

# Combine title and abstract embeddings
news_df['embeddings'] = news_df.apply(lambda x: np.concatenate((x['title_embedding'],
                                                                x['abstract_embedding'] if isinstance(x['abstract_embedding'], np.ndarray) else np.zeros_like(x['title_embedding']))),
                                      axis=1)

print("Embeddings loaded and merged.")

# Convert the embeddings to a NumPy array
X = np.array(news_df['embeddings'].tolist())

print("Embeddings converted to NumPy array.")

# Perform DBSCAN clustering for different epsilon values
max_epsilon = 10.0
step_size = 0.3
silhouette_scores = []

start_time = time.time()

for i, eps in enumerate(np.arange(0.5, max_epsilon + step_size, step_size)):
    dbscan = DBSCAN(eps=eps, min_samples=2)
    labels = dbscan.fit_predict(X)
    
    # Skip if all points are labeled as noise (-1)
    if len(set(labels)) <= 1:
        continue
    
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
    
    elapsed_time = time.time() - start_time
    print(f"Iteration {i+1}/{len(np.arange(0.1, max_epsilon + step_size, step_size))}, Epsilon: {eps:.1f}, Silhouette Score: {silhouette_avg:.3f}, Elapsed Time: {elapsed_time:.2f}s")

print("DBSCAN clustering completed.")

if silhouette_scores:
    # Find the best epsilon value based on silhouette score
    best_eps_index = np.argmax(silhouette_scores)
    best_eps = (best_eps_index + 1) * step_size
    print(f"Best epsilon value: {best_eps:.1f}")

    # Perform DBSCAN with the best epsilon value and adjusted parameters
    dbscan = DBSCAN(eps=best_eps, min_samples=10)
    labels = dbscan.fit_predict(X)
    print("DBSCAN clustering with best epsilon value completed.")

    # Plot epsilon vs. silhouette score
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=np.arange(0.1, max_epsilon + step_size, step_size)[:len(silhouette_scores)], y=silhouette_scores, marker='o')
    plt.title("Epsilon vs. Silhouette Score", fontsize=16)
    plt.xlabel("Epsilon", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Assign labels to the clusters based on their characteristics or content
    cluster_labels = {}
    for label in np.unique(labels):
        if label == -1:
            cluster_labels[label] = "Noise"
        else:
            cluster_points = news_df[labels == label]
            cluster_labels[label] = f"Cluster {label}" 

    # Plot the clusters using t-SNE with meaningful labels
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=cluster_labels[label])
    plt.title("DBSCAN Clustering Visualization (t-SNE)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Cluster", loc="best", fontsize=12)
    plt.show()

    # Print cluster information with meaningful labels
    for label in np.unique(labels):
        if label == -1:
            print(f"Noise points: {np.sum(labels == -1)}")
        else:
            print(f"{cluster_labels[label]}: {np.sum(labels == label)} points")
else:
    print("No valid epsilon values found. All data points are labeled as noise.")
