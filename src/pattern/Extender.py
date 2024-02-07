# Coded By Soroush Omidvartehrani


from sklearn.cluster import KMeans
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances_argmin_min
import time
import sys
import copy

# baseline
from Transformation.Blocks import *
from Transformation.Pattern import Pattern as RuleG
from Transformation.Blocks.SubstrG import SubstrG
from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock as LiteralG
from Transformation.Blocks.SplitG import SplitG
from Transformation.Blocks.SplitSubstrG import SplitSubstrG






class Rule:
    #units = []
    #coverage = -1
    #length = -1
    #i = -1
    #iterative = False

    def __init__(self):
        self.units = []
        self.parts = []
        self.partitioning = []
        self.iterative = False
        self.iteration = 0
        return

    def add(self, *arguments):
        parts = []
        default_part = Part(arguments, 'N', False)
        parts.append(default_part)
        self.parts = parts
        return

    def to_str(self):
        str = "["

        if len(self.parts) > 0:
            for part in self.parts:
                str += part.to_str()
                str += ", "
            str = str[:-2]

        str += "]"

        return str


class Unit:
    def __init__(self, name, start_index, end_index):
        self.name = name
        self.start_index = start_index
        self.end_index = end_index
        self.i = -1
        self.iterative = False
        self.type = "N"
        self.performed = False


class Part:
    def __init__(self, units, type, iterative):
        self.units = units
        self.type = type
        self.i = -1
        self.iterative = iterative

    def to_str(self):
        str = "["
        for unit in self.units:
            str += unit.to_str()
            str += ", "
        str = str[:-2]
        str += "]"
        str += self.type
        return str
    

class SubStr(Unit):
    def __init__(self, start_index, end_index):
        super().__init__("substr", start_index, end_index)

    def to_str(self):
        return (self.name + "(" + self.start_index + ", "+self.end_index+")")


class Split(Unit):
    delimiter = ''

    def __init__(self, delimiter, start_index, end_index):
        super().__init__("split", start_index, end_index)
        self.delimiter = delimiter

    def to_str(self):
        return (self.name + "(\'" + self.delimiter + "\', " +
                self.start_index + ", "+self.end_index+")")


class SplitSubStr(Unit):
    delimiter = ''
    part = -1

    def __init__(self, delimiter, part, start_index, end_index):
        super().__init__("splitsubstr", start_index, end_index)
        self.delimiter = delimiter
        self.part = part

    def to_str(self):
        return (self.name + "(" + self.delimiter + ", " + self.part +
                ", " + self.start_index + ", " + self.end_index+")")


class Literal(Unit):
    str = ""

    def __init__(self, str):
        super().__init__("literal", None, None)
        self.str = str

    def to_str(self):
        return (self.name + "(\'" + self.str + "\')")


def greedy_search(initial_rule):

    current_state = initial_rule
    current_score = estimated_coverage(current_state)

    while True:
        best_child = None
        best_score = current_score

        # I terate over all possible children of the current state
        for child in generalize(current_state):
            # Compute the score of the child
            child_score = estimated_coverage(child)

            # Check if the child is better than the current state
            if child_score > best_score:
                best_child = child
                best_score = child_score

        # If no better child was found, return the current state
        if best_child is None:
            return current_state

        # Otherwise, update the current state and score
        current_state = best_child
        current_score = best_score


def generalize(rule):

    new_rules = []

    def sub_lists(rule):
        l = rule.units
        lists = [[]]
        for i in range(len(l) + 1):
            for j in range(i):
                lists.append(l[j: i])
        return lists

    def powerset(s):
        x = len(s)
        masks = [1 << i for i in range(x)]
        for i in range(1 << x):
            yield [ss for mask, ss in zip(masks, s) if i & mask]

    def subset_indices(rule):
        n = len(rule)
        subsets = []
        for i in range(1 << n):  # Generate all possible combinations of indices
            subset_indices = []

            for j in range(n):
                if i & (1 << j):  # Check if the jth bit is set in the binary representation of i
                    subset_indices.append(j)

            subsets.append(subset_indices)
        return subsets

    def get_all_partitions(rule):
        arr = rule.parts[0].units
        partitions = []
        current_partition = []

        def backtrack(start):
            if start == len(arr):
                partitions.append(current_partition[:])
                return

            for end in range(start, len(arr)):
                current_partition.append(arr[start:end + 1])
                backtrack(end + 1)
                current_partition.pop()

        backtrack(0)
        return partitions

    def remove_unit(rule):
        new_rules = []
        # replace u with i in the if part of inner for
        duplicate_parts = subset_indices(rule)
        for duplicate_part in duplicate_parts:
            new_rule = Rule()
            new_rule.iterative = True
            new_rule.iteration = rule.iteration
            # new_rule.add(rule.units)
            for i, u in enumerate(rule.units):
                if i not in duplicate_part:
                    unit = copy.copy(rule.units[i])
                    new_rule.add(unit)  # (rule.units[i])
                    new_rule.units[-1].type = "N"
                else:
                    unit = copy.copy(rule.units[i])
                    new_rule.add(unit)  # (rule.units[i])
                    new_rule.units[-1].iterative = True
                    new_rule.units[-1].type = "R"
            new_rules.append(new_rule)

        return new_rules

    def get_subarray_indices(rule):
        arr = rule.units
        indices = []  # List to store start and end indices of subarrays

        for length in range(1, len(arr) + 1):
            for start in range(len(arr) - length + 1):
                end = start + length - 1
                indices.append((start, end))

        return indices

    def partitioning(rule):
        new_rules = []
        ready_to_partition_rules = get_all_partitions(rule)
        for ready_to_partition_rule in ready_to_partition_rules:

            possibilities = subset_indices(ready_to_partition_rule)

            new_rule = Rule()
            new_rule.iterative = True
            new_rule.iteration = rule.iteration

            new_parts = []
            for u in ready_to_partition_rule:
                new_part = Part(u, "N", True)
                new_parts.append(new_part)

            new_rule.parts = new_parts
            new_rules.append(new_rule)

        return new_rules

    def duplicate_part(rule):
        new_rules = []
        # duplicate_parts = subset_indices(rule)
        duplicate_parts = sub_lists(rule)
        # print(duplicate_parts)
        for duplicate_part in duplicate_parts:
            new_rule = Rule()
            new_rule.iterative = True
            new_rule.iteration = rule.iteration
            # new_rule.add(rule.units)
            for i, u in enumerate(rule.units):
                if u not in duplicate_part:
                    unit = copy.copy(rule.units[i])
                    new_rule.add(unit)  # (rule.units[i])
                    new_rule.units[-1].type = "N"
                else:
                    unit = copy.copy(rule.units[i])
                    new_rule.add(unit)  # (rule.units[i])
                    new_rule.units[-1].iterative = True
                    new_rule.units[-1].type = "D"
            new_rules.append(new_rule)

        return new_rules

        def duplicate_unit(rule):
            new_rules = []
            #duplicate_parts = subset_indices(rule)
            duplicate_parts = sub_lists(rule)
            # print(duplicate_parts)
            for duplicate_part in duplicate_parts:
                new_rule = Rule()
                new_rule.iterative = True
                new_rule.iteration = rule.iteration
                # new_rule.add(rule.units)
                for i, u in enumerate(rule.units):
                    if u not in duplicate_part:
                        unit = copy.copy(rule.units[i])
                        new_rule.add(unit)  # (rule.units[i])
                        new_rule.units[-1].type = "N"
                    else:
                        unit = copy.copy(rule.units[i])
                        new_rule.add(unit)  # (rule.units[i])
                        new_rule.units[-1].iterative = True
                        new_rule.units[-1].type = "D"
                new_rules.append(new_rule)

            return new_rules


    partitioned_rules = partitioning(rule)

    # for partitioned_rule in partitioned_rules:
    #    duplicate_part(partitioned_rule)

    #new_rules += remove_unit(rule)
    #new_rules += duplicate_unit(rule)
    new_rules += partitioning(rule)

    return new_rules


def rule_convertor(rule, itr = 2, punctuations=[' '], mode="duplicating"):

    def unit_convertor(unit, i, punctuation=punctuations[0]):

        def process_string(input_string, number):
            character = input_string[0]
            if len(input_string) == 1:
                input_string += "+0"
            try:
                value = int(input_string[1:])
                result = value + number
                if (character == "e" and result > 0) or (character == "s" and result < 0):
                    result = 0
                if result != 0:
                    return f"{character}{result:+}"
            except ValueError:
                pass
            return character

        if (unit.name == "split"):
            u = SplitG(unit.delimiter, process_string(
                unit.start_index, i), process_string(unit.end_index, i))
        elif (unit.name == "substr"):
            u = SplitSubstrG(punctuation, process_string("s", i),
                             unit.start_index, unit.end_index)
        elif (unit.name == "splitsubstr"):
            u = SplitSubstrG(unit.delimiter, process_string(unit.part, i),
                             unit.start_index, unit.end_index)
        elif (unit.name == "literal"):
            u = LiteralG(unit.str)

        return u


    def generator(rule, i, ongoing_rule=[], result=[], mode=mode):
        if ongoing_rule:
            if mode != "ordering":
                result.append(ongoing_rule.copy())
            else:
                if len(ongoing_rule) == len(rule.parts):
                    result.append(ongoing_rule.copy())

        for part in rule.parts:
            if mode == "duplicating":
                condition = ongoing_rule.count(part) < i and (
                    not ongoing_rule or rule.parts.index(part) >= rule.parts.index(ongoing_rule[-1]))
            elif mode == "ordering":
                i = 1
                condition = ongoing_rule.count(part) < i
            elif mode == "removing":
                i = 1
                condition = ongoing_rule.count(part) < i and (
                    not ongoing_rule or rule.parts.index(part) >= rule.parts.index(ongoing_rule[-1]))
            if (condition):
                # Check if the element has appeared less than i times
                ongoing_rule.append(part)
                # Recursively generate the remaining elements
                generator(rule, i, ongoing_rule, result)
                # Remove the last element to backtrack and explore other possibilities
                ongoing_rule.pop()

        return result

    def find_i(rule):
        units = []
        for part in rule:
            for unit in part.units:
                units.append(unit)

        seen_counts = {}
        result = []
        for element in units:
            if element in seen_counts:
                seen_counts[element] += 1
            else:
                seen_counts[element] = 0
            result.append(seen_counts[element])

        return result

    def convertor(generated_rules):
        converted_rules = set()
        for generated_rule in generated_rules:
            i = find_i(generated_rule)
            # print(i)
            temp_units = []

            for part in generated_rule:
                for unit in part.units:
                    # if unit.name != "substr":
                    temp_units.append(unit)

            converted_units = []
            unfinished_rules = []
            unfinished_rules.append(converted_units)

            for unit in temp_units:
                if unit.name == "substr":
                    temp_rules = [copy.deepcopy(
                        arr) for arr in unfinished_rules for _ in range(len(punctuations))]
                    unfinished_rules = temp_rules

                    for unfinished_rule in unfinished_rules:
                        idx = unfinished_rules.index(
                            unfinished_rule) % len(punctuations)
                        u = unit_convertor(
                            unit, i[len(unfinished_rule)], punctuations[idx])
                        unfinished_rule.append(u)
                else:
                    for unfinished_rule in unfinished_rules:
                        u = unit_convertor(unit, i[len(unfinished_rule)])
                        unfinished_rule.append(u)

            for unfinished_rule in unfinished_rules:
                generated_rule = RuleG(unfinished_rule)
                converted_rules.add(generated_rule)

        return converted_rules

    generated_rules = generator(rule, itr)
    converted_rules = convertor(generated_rules)
    return converted_rules



def rule_preprocessing(rule, itr = 2):

    temp_rule = Rule()

    def unit_convertor(unit):
        if (unit.NAME == "SPLITG"):
            u = Split(unit._splitter, unit._index_start, unit._index_end)
        elif (unit.NAME == "SUBSTRG"):
            u = SubStr(unit._start, unit._end)
        elif (unit.NAME == "SPLT_SUB_G"):
            u = SplitSubStr(unit._splitter, unit._index, unit._start, unit._end)
        elif (unit.NAME == "LITERAL"):
            u = Literal(unit.text)

        return u
    
    unfinished_rule = []

    for unit in rule.blocks:
        unfinished_rule.append(unit_convertor(unit))

    temp_rule.add(*unfinished_rule)
    temp_rule.iteration = itr
    
    return temp_rule


# -------------------------------Scoring Part-------------------------------#


def read_csv_file(addr, source, target):
    df = pd.read_csv(addr, usecols=[source, target])
    return df

def find_punctuations(df):
    punctuations = [' ', '¿']
    df['concatenation'] = df.astype(str).apply(' '.join, axis=1)
    for index, row in df.iterrows():
        temp = row['concatenation']
        for i in temp:
            if i in string.punctuation and i not in punctuations:
                punctuations.append(i)
    return punctuations

def similarity_generator(seperators, source, target):

    def units_per_string(c, str):
        units = str.split(c)
        return len(units)

    similarity = []
    for s in seperators:
        similarity.append(units_per_string(s, source))
        similarity.append(units_per_string(s, target))
    return similarity

def kmeans(df):
    df_for_kmeans = pd.DataFrame(df.similarity.tolist(), index=df.index)
    # print(df_for_kmeans.head())
    mat = df_for_kmeans.values
    # print(mat)

    # plot
    # plt.plot(range(2, 5), distortions, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # plt.show()

    # finding best k
    best_k = 0
    best_silhouette = -1
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42,
                        n_init=10, max_iter=300, tol=1e-04)
        cluster_labels = kmeans.fit_predict(mat)
        silhouette_avg = silhouette_score(mat, cluster_labels)
        # print(f"For n_clusters = {k}, the average silhouette score is {silhouette_avg:.2f}")
        if silhouette_avg > best_silhouette:
            best_k = k
            best_silhouette = silhouette_avg

    def clustering(df, X, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        cluster_labels = kmeans.fit_predict(mat)
        silhouette_avg = silhouette_score(mat, cluster_labels)
        df['cluster'] = kmeans.labels_
        print(
            f"For n_clusters = {k}, the average silhouette score is {silhouette_avg:.2f}")

        unique_labels = set(kmeans.labels_)
        for label in unique_labels:
            print(f"Cluster {label}: \n{df.loc[df['cluster'] == label]}")
        return kmeans

    kmeans = clustering(df, mat, best_k)

    # print(df.head())

    return kmeans, df, mat

def cluster_sampling(alg, df, mat):

    def find_centers(kmeans, mat):
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, mat)
        print(f"\nRepresentatives:")
        for val in closest:
            print(f"\nCluster {df.loc[val].cluster}: \n{df.loc[val]}")
        return closest

    def cluster_size(kmeans, mat):
        # find the number of items in each cluster
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        cluster_weights = dict(zip(unique, counts/mat.shape[0]))

        print(dict(zip(unique, counts)))
        print(cluster_weights)
        return cluster_weights

    def representative_weight(dict1, arr):
        tuples = []
        tuples = []

        # Loop through the indices of the array
        for i in range(len(arr)):
            # Get the value from the array and the dictionary using the index
            value1 = arr[i]
            value2 = dict1[i]

            # Create a tuple from the values and append it to the list
            tuples.append((value2, value1))

        # Return the list of tuples
        return tuples

    kmeans = alg
    centers = find_centers(kmeans, mat)
    print(centers)
    weights = cluster_size(kmeans, mat)
    print(weights)

    rep_w = representative_weight(centers, weights)
    print("Representatives:")
    print(rep_w)

    return rep_w

def uniform_sampling(alg, df, mat):
    kmeans = alg
    n = kmeans.n_clusters
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
            value2 = 1/len(arr)

            # Create a tuple from the values and append it to the list
            tuples.append((value1, value2))

        # Return the list of tuples
        return tuples

    # Return the sampled indices as a list
    print("Representatives:")
    sampled_indices_weighted = representative_weight(sampled_indices)
    print(sampled_indices_weighted)
    return list(sampled_indices)

def estimated_coverage(path):
    # todo
    return


# ordering, duplicating, removing
def generalizer(rule=None, itr=2, path=None, src=None, trg=None, mode="duplicating"):

    # find punctuation
    df = read_csv_file(path, src, trg)
    punc = find_punctuations(df)
    # print("Punctuations:")
    # print(punc)

    # input rule
    #print("Input rule:")
    #print(rule)

    rules = set()

    preprocessed_rule = rule_preprocessing(rule, itr)
    generalized_rules = generalize(preprocessed_rule)

    #print("\nGeneralized rules: ")
    #for generalized_rule in generalized_rules:
        #print("Rule " + str(generalized_rules.index(generalized_rule)+1) + ":")
        #print(generalized_rule.to_str())

    #print("\nGeneralized rules (converted version): ")
    for generalized_rule in generalized_rules:
        #print("Rule " + str(generalized_rules.index(generalized_rule)+1) + ":")
        coneverted_rules = rule_convertor(
            generalized_rule, itr, punctuations=punc, mode=mode)  # ordering, duplicating, removing
        for coneverted_rule in coneverted_rules:
            rules.add(coneverted_rule)
            # print(coneverted_rule)
        #print(" ")


    return rules

def main():
    # start
    start_time = time.time()

    print("Python version:")
    print(sys.version+"\n")

    path = "/Users/Soroush/Desktop/Project/Rule Generalization/Code/data/autojoin-Benchmark/california govs 1/ground truth.csv"
    src = "source-Governor's Name"
    trg = "target-Name"


    input_rule = RuleG([
    #SubstrG('s+1', 'e-1'),
    LiteralG("Txt"),
    #SplitG(" ", "s-0", "e-1"),
    SplitSubstrG(" ", "e-1", "s-0", "e-1"),
    ])

    rules = generalizer(input_rule, 2,path,src, trg,"ordering")
    print("Number of generated rules= " + str(len(rules)) )

    # end
    end_time = time.time()
    print("\nRuntime = " + str(end_time - start_time))


if __name__ == "__main__":
    main()
