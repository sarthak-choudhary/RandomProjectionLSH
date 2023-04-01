from datasketch import MinHashLSHForest, MinHash
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score

# Generate some high-dimensional data

x, _ = make_blobs(n_samples=10000, centers=2, n_features=10, random_state=42)

# Initialize MinHash object
minhashes = []
num_perm = 128

for i in range(len(x)):
    m = MinHash(num_perm=num_perm)
    for d in x[i]:
        m.update(str(d).encode('utf8'))
    
    minhashes.append(m)
    
num_trees = 10
forest = MinHashLSHForest(num_perm=num_perm, l=num_trees)

# Add data points to the forest
for i, data_point in enumerate(x):
    # Get MinHash signature for each data point
    sign = minhashes[i]
    
    # Add data point to the forest
    forest.add(str(i), sign)
    

# Index the LSHForest
forest.index()

# Retrieve the hash codes
hash_codes = np.array([mh.digest() for mh in minhashes])

# Perform clustering on the hash codes
kmeans= KMeans(n_clusters=2)
clusters = kmeans.fit(hash_codes)
labels = clusters.labels_

original_kmeans = KMeans(n_clusters=2)
original_clusters = original_kmeans.fit(x)
original_labels = original_clusters.labels_

ami = adjusted_mutual_info_score(original_labels, labels)
print('AMI: ', ami)