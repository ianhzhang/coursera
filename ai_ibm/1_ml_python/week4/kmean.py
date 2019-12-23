import random 
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 

# 1. Generate data
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# 2. Setup Model and training the data

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)

# 3. Results:
print(k_means.labels_)  # it is a list corresponding to chech cluster
print("=========================================")
print(k_means.cluster_centers_)     # 4 cluster centers