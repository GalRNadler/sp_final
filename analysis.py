import sys
import math
import mysymnmf
from sklearn.metrics import silhouette_score
import numpy as np

DEFAULT_ITER = 300
DEFAULT_EPSILON = 0.0001

class KmeanGroup:
    def __init__(self, vector):
        self.rep = vector
        self.members = []

    def calc_euclidean_dis_with_rep(self, x):
        return self.calc_euclidean_dis(x, self.rep)

    def calc_euclidean_dis(self, x, centroid):
        distance = 0
        for cor in range(len(centroid)):
            distance += math.pow((centroid[cor] - x[cor]), 2)
        return math.sqrt(distance)

    def add_member(self, vector):
        self.members.append(vector)

    def update_rep_and_check_accuracy(self):
        prev_rep = self.rep
        new_centroid = []
        for cor in range(len(self.rep)):
            sum_cor = 0
            for member in self.members:
                sum_cor += member[cor]
            if len(self.members) == 0:
                new_centroid.append(0)
            else:
                new_centroid.append(sum_cor/len(self.members))
        self.rep = new_centroid
        return self.calc_euclidean_dis(self.rep, prev_rep) < DEFAULT_EPSILON

    def clear_members_list(self):
        self.members = []


def initialize(k, datapoints):
    k_means = []
    for i in range(k):
        vector = datapoints[i]
        k_means.append(KmeanGroup(vector))
    return k_means


def update_and_check_centroids_accuracy(k_means):
    to_continue_iter = False
    for i in k_means:
        if not i.update_rep_and_check_accuracy():
            to_continue_iter = True
        i.clear_members_list()
    return to_continue_iter


def calculate_kmeans(k, datapoints):
    k_means = initialize(k, datapoints)
    current_iter = 0
    to_continue_iter = True
    kmeans_labels = np.array([0] * len(datapoints))

    while to_continue_iter and current_iter < DEFAULT_ITER:
        for i in range(len(datapoints)):
            vector = datapoints[i]
            relevant_k_mean_group = False
            min_dis = False
            for j in range(k):
                temp_cal = k_means[j].calc_euclidean_dis_with_rep(vector)
                if relevant_k_mean_group is False or temp_cal < min_dis:
                    min_dis = temp_cal
                    relevant_k_mean_group = k_means[j]
            relevant_k_mean_group.add_member(vector)
            kmeans_labels[i] = k_means.index(relevant_k_mean_group)
        to_continue_iter = update_and_check_centroids_accuracy(k_means)
        current_iter += 1
    return kmeans_labels


def initializeH(datapoints, k, W):
    np.random.seed(0)
    n = len(datapoints)
    m = np.mean(W)
    m_divide_k = m / float(k)
    high_level = 2 * math.sqrt(m_divide_k)
    matrix = np.random.uniform(0, high_level, size=(n, k)).tolist()
    return matrix


def format_data_points(file_name):
    datapoints = []
    with open(file_name, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            float_values = [float(val) for val in values]
            datapoints.append(float_values)
    return datapoints


def parse_input():
    k = sys.argv[1]
    file_name = sys.argv[2]
    datapoints = format_data_points(file_name)
    return int(k), datapoints


def calc_symnmf_labels(symNMF):
    labels = []
    for vector in symNMF:
        max_val = max(vector)
        center_idx = vector.index(max_val)
        labels.append(center_idx)
    return np.array(labels)


def main():
    k, datapoints = parse_input()
    kmeans_labels = calculate_kmeans(k, datapoints)
    kmeans_silhouette = format(silhouette_score(datapoints, kmeans_labels), '.4f')

    W = mysymnmf.norm(0, len(datapoints), len(datapoints[0]), datapoints)
    H = initializeH(datapoints, k, W)
    symNMF = mysymnmf.symnmf(k, len(datapoints), W, H, 1)
    sym_labels = calc_symnmf_labels(symNMF)
    nmf_silhouette = format(silhouette_score(datapoints, sym_labels), '.4f')

    final_calc = "nmf: {0}\nkmeans: {1}".format(nmf_silhouette, kmeans_silhouette)
    print(final_calc)


if __name__ == "__main__":
    main()