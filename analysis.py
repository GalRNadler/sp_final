import sys
from math import sqrt
from sklearn.metrics import silhouette_score
import numpy as np
import symnmfmodule

np.random.seed(0)

def euclidean_distance(vec1, vec2):
    d = 0
    for x1, x2 in zip(vec1, vec2):
        d_single = (x1 - x2) ** 2
        d += d_single
    return sqrt(d)


def to_number(num):
    try:
        return int(float(num))
    except:
        raise Exception()

def init_vector_list(input_data):
    vectors = []
    with open(input_data) as vectors_file:
        for line in vectors_file:
            vector = line.strip()
            if vector:
                vectors.append(list(float(point) for point in vector.split(",")))
    return vectors

def parse_input():
    if len(sys.argv) == 3:
        k = to_number(sys.argv[1])
        input_data = sys.argv[2]
        datapoints = init_vector_list(input_data)

        n = len(datapoints)
        d = len(datapoints[0])
        return k, datapoints, n, d
    else:
        raise Exception()

def find_closest_centroid(curr, k_centroids):
    closest_d = float("inf")
    closest_centroid = None

    for centroid in k_centroids.keys():
        distance = euclidean_distance(curr, centroid)
        if distance < closest_d:
            closest_d = distance
            closest_centroid = centroid
    return closest_centroid


def calculate_updated_centroid(centroid_vectors, d):
    num_vectors = len(centroid_vectors)

    updated_centroid = []
    for i in range(d):
        avg_point = (sum(v[i] for v in centroid_vectors)) / num_vectors
        updated_centroid.append(avg_point)
    return tuple(updated_centroid)


def assign_to_closest_centroid(datapoints, k_centroids, vectors_mapping):
    for curr_vect in datapoints:
        curr_vect_tuple = tuple(curr_vect)  # Convert the current vector to a tuple
        closest_centroid = find_closest_centroid(curr_vect_tuple, k_centroids)
        prev_centroid = vectors_mapping[curr_vect_tuple]  # find prev mapping
        if closest_centroid != prev_centroid:
            if prev_centroid is not None:
                k_centroids[prev_centroid].remove(curr_vect)  # remove the curr from the prev centroid
            vectors_mapping[curr_vect_tuple] = closest_centroid  # add to new mapping
            k_centroids[closest_centroid].append(curr_vect)  # add the curr to its closest centroid


def convert_centroids_to_labels(datapoints, k_means_centroids):
    labels = []
    centroid_map = {centroid: i for i, centroid in enumerate(k_means_centroids)}
    for point in datapoints:
        for centroid, points in k_means_centroids.items():
            if point in points:
                labels.append(centroid_map[centroid])
                break
    return labels

def k_means(k, d, datapoints):
    e = 0.0001
    max_iter = 300
    k_centroids = {tuple(datapoints[i]): [datapoints[i]] for i in range(k)}
    vectors_mapping = {tuple(vector): tuple(vector) if tuple(vector) in k_centroids.keys() else None for vector in datapoints}
    i = 0
    delta_miu = float("inf")
    while delta_miu >= e or i < max_iter:
        assign_to_closest_centroid(datapoints, k_centroids, vectors_mapping)
        new_k_centroids = dict()
        for curr_centroid in k_centroids:
            updated_centroid = calculate_updated_centroid(k_centroids[curr_centroid], d)
            new_k_centroids[updated_centroid] = k_centroids[curr_centroid]  # change to new centroid
            for vector in new_k_centroids[updated_centroid]:
                vectors_mapping[tuple(vector)] = updated_centroid
            delta_miu = min(euclidean_distance(curr_centroid, updated_centroid), delta_miu)
        k_centroids = new_k_centroids

        i += 1

    return convert_centroids_to_labels(datapoints, k_centroids)

def init_h(n, k, W):
    mean_w = np.mean(W)
    constant_term = 2 * sqrt(mean_w / k)
    H = np.random.uniform(0, high=constant_term, size=(n, k))
    return H.tolist()

def main():
    try:
        k, datapoints, n, d = parse_input()
        k_means_labels = k_means(k, d, datapoints)
        
        W = symnmfmodule.norm(0, n, d, datapoints)
        H = init_h(n, k, W)
        symNMF = np.array(symnmfmodule.symnmf(k, n, W, H, 1))
        sym_labels = symNMF.argmax(axis=1)
    
        print("nmf: %.4f" % silhouette_score(datapoints, sym_labels))
        print("kmeans: %.4f" % silhouette_score(datapoints, k_means_labels))
    
    except Exception:
        print("An Error Has Occurred")

if __name__ == "__main__":
    main()