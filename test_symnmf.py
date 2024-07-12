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
    input_matrix_list = [list(row) for row in input_matrix]
    H_list = [list(row) for row in H]

    # Ensure k, num_points, and num_features are integers
    k = int(k)
    num_points = int(num_points)
    num_features = int(num_features)

    # Analysis flag (set to 1 to enable analysis)
    analysis = 1

    # Debugging information
    print(f"k: {k}, num_points: {num_points}, num_features: {num_features}")
    print(f"input_matrix_list: {input_matrix_list}")
    print(f"H_list: {H_list}")
    print(f"analysis: {analysis}")

    # Verify types of all arguments
    print(f"type(k): {type(k)}, type(num_points): {type(num_points)}, type(num_features): {type(num_features)}")
    print(f"type(input_matrix_list): {type(input_matrix_list)}, type(H_list): {type(H_list)}, type(analysis): {type(analysis)}")

    # Check elements inside the list
    print(f"type(input_matrix_list[0]): {type(input_matrix_list[0])}, type(input_matrix_list[0][0]): {type(input_matrix_list[0][0])}")
    print(f"type(H_list[0]): {type(H_list[0])}, type(H_list[0][0]): {type(H_list[0][0])}")

    # Run SYMNMF algorithm
    try:
        result = symnmfmodule.symnmf(k, num_points, input_matrix_list, H_list, analysis)
        # Print the result
        print("Result of SYMNMF:")
        for row in result:
            print(row)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
