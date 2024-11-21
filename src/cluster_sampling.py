from sklearn.cluster import KMeans
import pandas as pd
import os
import string
import pathlib
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import datetime
import warnings
warnings.filterwarnings("ignore")

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
table_rel_path = "data/autojoin-Benchmark/sharif username to email/ground truth.csv" #"data/FlashFill/dr-name-small/ground truth.csv" #"data/autojoin-Benchmark/us presidents 6/ground truth.csv"
PATH = os.path.join(BASE_PATH, table_rel_path)

addr = PATH
src = "source-Username"
trg = "target-Email"
alg = "kmeans"
log = True

MIN_SAMPLE = 2
MAX_RUN = 5

class Transformation:
    value = ""
    coverage = 0
    similarities = []

    def __init__(self, value, coverage):
        self.value = value
        self.coverage = coverage

    def check(source, target):
        return False

    def update(value):
        self.value = value
        return

def units_per_string(c, s):
    units = str(s).split(c)
    return len(units)

def read_csv_file(addr, source, target):
    df = pd.read_csv(addr, usecols=[source, target], dtype=str)
    return df

def find_punctuations(df):
    punctuations = []
    df['concatenation'] = df.apply(' '.join, axis=1) 
    for index, row in df.iterrows():
        temp = row['concatenation']
        for i in temp:
            if i in string.punctuation and i not in punctuations:
                punctuations.append(i)
    return punctuations

def similarity_generator(seperators, source, target):
    similarity = []
    for s in seperators:
        similarity.append(units_per_string(s, source))
        similarity.append(units_per_string(s, target))
    return similarity


    return

def kmeans(df, log=True):
    df_for_kmeans = pd.DataFrame(df.similarity.tolist(), index=df.index)

    mat = df_for_kmeans.values # = similarity column

    # finding best k
    best_k = 2
    best_silhouette = -1
    for k in range(2, min(MAX_RUN, df_for_kmeans.shape[0])):
        kmeans = KMeans(n_clusters=k, random_state=1) #,n_init=MAX_RUN, max_iter=300, tol=1e-04)
        cluster_labels = kmeans.fit_predict(mat)
        # silhouette_avg = silhouette_score(mat, cluster_labels)
        if all(kmeans.labels_ == 0):
            silhouette_avg = -0.9999
        else:
            silhouette_avg = silhouette_score(mat, cluster_labels)
        # print(f"For n_clusters = {k}, the average silhouette score is {silhouette_avg:.2f}")
        if silhouette_avg > best_silhouette:
            best_k = k
            best_silhouette = silhouette_avg

    def clustering(df, X, k):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(X)

        df['cluster'] = kmeans.labels_
        #print(f"For n_clusters = {k}, the average silhouette score is {silhouette_avg:.2f}")

        unique_labels = set(kmeans.labels_)

        
        if (log == True):
            for label in unique_labels:
                print(f"Cluster {label}: \n{df.loc[df['cluster'] == label]}")
        
        return kmeans

    kmeans = clustering(df, mat, best_k)

    # print(df.head())

    return kmeans, df, mat


def cluster_sampling(alg, df, mat):

    def find_smaples(kmeans, mat):
        #closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, mat)

        labels = kmeans.labels_
        cluster_indices = {}

        # Iterate through data points and populate the dictionary
        for i, label in enumerate(labels):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(i)
        
        result_indices = []
        for cluster, indices in cluster_indices.items():
            if len(indices) >= MIN_SAMPLE:
                result_indices.extend(indices[:MIN_SAMPLE])
            else:
                result_indices.extend(indices)

        # print(result_indices)
        # print(f"\nRepresentatives:")
        # for val in closest:
        # print(f"\nCluster {df.loc[val].cluster}: \n{df.loc[val]}")
        return result_indices

    kmeans = alg

    samples = find_smaples(kmeans, mat)
    #print("samples: " + str(samples))

    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    total_rows = len(labels)

    # Calculate the weight of each row based on the cluster assignments
    rep_w = [(index, cluster_counts[labels[index]] / total_rows) for index in samples]


    if (len(rep_w) <= MIN_SAMPLE): # kmeans is not effective
        #print("kmeans is NOT effective")
        rep_w = uniform_sampling(df,min(MIN_SAMPLE,len(df.index)))
    #else:
        #print("kmeans is effective")
        #rep_w = uniform_sampling(df,len(rep_w))

    return rep_w


def uniform_sampling(df, sample_size):
    #kmeans = alg
    n = sample_size #kmeans.n_clusters
    # Get the total number of rows in the DataFrame
    num_rows = df.shape[0]

    # Generate a random permutation of the row indices
    permuted_indices = np.random.permutation(num_rows)

    # Sample n indices from the permutation
    sampled_indices = permuted_indices[:n]

    def representative_weight(arr):
        tuples = []
        tuples = []

        # Loop through the indices of the array
        for i in range(len(arr)):
            # Get the value from the array and the dictionary using the index
            value1 = arr[i]
            value2 = 1 / len(arr)

            # Create a tuple from the values and append it to the list
            tuples.append((value1, value2))

        # Return the list of tuples
        return tuples

    # Return the sampled indices as a list
    sampled_indices_weighted = representative_weight(sampled_indices)
    
    return list(sampled_indices_weighted)


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    df = read_csv_file(addr, src, trg)

    print("Read CSV - Time: " + str((datetime.datetime.now() - start_time).seconds))

    # find punctuations
    seperators = find_punctuations(df)
    seperators.append(" ")

    print("Find Punctuations - Time: " + str((datetime.datetime.now() - start_time).seconds))

    # create similarity column
    df['similarity'] = df.apply(
        lambda x: similarity_generator(seperators, x[0], x[1]), axis=1)
    
    print("Create Similarity Column - Time: " + str((datetime.datetime.now() - start_time).seconds))

    if alg == "kmeans":
        km, df, mat = kmeans(df, log)
        representatives = cluster_sampling(km, df, mat)
        print(representatives)
    
    print("Kmeans - Time: " + str((datetime.datetime.now() - start_time).seconds))
