import mysymnmf as symnmfmodule
import numpy as np

def load_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [list(map(float, line.strip().split(','))) for line in lines]
    return np.array(data)

def main():
    # Load the small input matrix
    input_matrix = load_input('input_1.txt')

    # Assume k = 2 for this small test case
    k = 2
    num_points = input_matrix.shape[0]
    num_features = input_matrix.shape[1]

    # Initialize H matrix with random values
    H = np.random.rand(num_points, k)

    # Convert numpy arrays to lists of lists
    input_matrix_list = input_matrix.tolist()
    H_list = H.tolist()

    # Ensure k, num_points, and num_features are integers
    k = int(k)
    num_points = int(num_points)
    num_features = int(num_features)

    # Analysis flag (set to 1 to enable analysis)
    analysis = 1

    # Run SYMNMF algorithm
    result = symnmfmodule.symnmf(k, num_points, input_matrix_list, H_list, analysis)

    # Print the result
    print("Result of SYMNMF:")
    for row in result:
        print(row)

if __name__ == "__main__":
    main()
