import random 
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 

# 1. get data
cust_df = pd.read_csv("Cust_Segmentation.csv")
df = cust_df.drop('Address', axis=1)

# 2. Preprocessing
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)

# 3. Setup Model and training the data

k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means.fit(X)


# 4. Results:

print(k_means.labels_)  # it is a list corresponding to chech cluster
print("=========================================")
print(k_means.cluster_centers_)     # 4 cluster centers

# append lable to original dataframe
df["Cluster"] = k_means.labels_
