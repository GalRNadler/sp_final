import sys
import math
import numpy as np
import mysymnmf

OPTIONS = ["symnmf", "sym", "ddg", "norm"]
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
    goal = sys.argv[2]
    file_name = sys.argv[3]

    datapoints = format_data_points(file_name)
    return int(k), goal, datapoints

def initilize_H(datapoints, k, W):
    np.random.seed(0)
    n = len(datapoints)
    m = np.mean(W)
    m_divide_k = m/ float(k)
    high_level = 2 * math.sqrt(m_divide_k)
    matrix = np.random.uniform(0, high_level, size = (n, k)).tolist()
    return matrix

def logic(datapoints, k, goal):
    vNum = len(datapoints)
    vSize = len(datapoints[0])
    if goal == "symnmf":
        needToPrintNorm = 0
        W = mysymnmf.norm(needToPrintNorm, vNum, vSize ,datapoints)
        H = initilize_H(datapoints, k, W)
        res = mysymnmf.symnmf(k, vNum, W, H, 0)
    elif goal == "sym":
        res = mysymnmf.sym(vNum, vSize, datapoints)
    elif goal == "ddg":
        res = mysymnmf.ddg(vNum, vSize, datapoints)
    elif goal == "norm":
        needToPrintNorm = 1
        res = mysymnmf.norm(needToPrintNorm, vNum, vSize,datapoints)
    return

def print_matrix(matrix):
    string = ""
    for row in matrix:
        for vector in row:
            string += "{:.4f}".format(vector) + ","
        string = string[:-1]
        string += "\n"
    print(string)
    print('')

def main():
    try:
        k, goal, datapoints = parse_input()
        logic(datapoints, k, goal)
    except:
        raise RuntimeError("An Error Has Occurred")

if __name__ == "__main__":
    main()