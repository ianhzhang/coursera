import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 

X, y = make_blobs(n_samples=1500, 
                  centers=[[4,3], [2,-1], [-1,4]], 
                  cluster_std=0.5)
    
# Standardize features by removing the mean and scaling to unit variance
X = StandardScaler().fit_transform(X)
print(X)
print(y)
db = DBSCAN(eps=0.3, min_samples=7).fit(X)
print(db.labels_)