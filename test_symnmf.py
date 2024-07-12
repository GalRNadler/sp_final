import symnmfmodule
import numpy as np

def load_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [list(map(float, line.split())) for line in lines]
    return np.array(data)

def main():
    # Load the small input matrix
    input_matrix = load_input('small_input.txt')

    # Assume k = 2 for this small test case
    k = 2
    num_points = input_matrix.shape[0]
    num_features = input_matrix.shape[1]

    # Initialize H matrix with random values
    H = np.random.rand(num_points, k)

    # Run SYMNMF algorithm
    result = symnmfmodule.symnmf(k, num_points, input_matrix.tolist(), H.tolist())

    # Print the result
    print("Result of SYMNMF:")
    for row in result:
        print(row)

if __name__ == "__main__":
    main()
