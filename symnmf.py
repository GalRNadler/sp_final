import sys
from math import sqrt
import numpy as np
import symnmfmodule

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
    k = to_number(sys.argv[1])
    goal = sys.argv[2]
    input_data = sys.argv[3]

    d_points = init_vector_list(input_data)
    n = len(d_points)
    d = len(d_points[0])
    return d_points, k, goal, n, d

def init_h(n, k, W):
    np.random.seed(0)
    mean_w = np.mean(W)
    constant_term = 2 * sqrt(mean_w / k)
    H = np.random.uniform(0, high=constant_term, size=(n, k))
    return H.tolist()

def logic(d_points, k, goal, n, d):
    if goal == "symnmf":
        W = symnmfmodule.norm(0, n, d ,d_points)
        H = init_h(n, k, W)
        symnmfmodule.symnmf(k, n, W, H, 0)
    elif goal == "similarity_matrix":
        symnmfmodule.similarity_matrix(n, d, d_points)
    elif goal == "diagonal_matrix":
        symnmfmodule.diagonal_matrix(n, d, d_points)
    elif goal == "norm_matrix":
        symnmfmodule.norm_matrix(1, n, d, d_points)
    else:
        raise Exception()

def main():
    try:
        d_points, k, goal, n, d = parse_input()
        logic(d_points, k, goal, n, d)
    except Exception as e:
        print("An Error Has Occurred")

if __name__ == "__main__":
    main()
