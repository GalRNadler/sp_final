void free_matrix_memory(double **matrix, int vNum);
void print_matrix(double **datapoints, int vNum, int vSize);
double sum_vector_coordinates(double *v1, int vSize);
double calculate_squared_euclidean_distance(double *v1, double *v2, int vSize);
double **calc_similarity_matrix(int vNum, int vSize, double **datapoints);
double **init_matrix(int vNum, int vSize);
double **calc_diagonal_matrix(int vNum, int vSize, double **datapoints);
double **calc_normalized_similarity_matrix(int vNum, int vSize, double **datapoints);
double **calc_symnmf(int k, int vNum, double **norm_matrix, double **H);
int has_converged(int k, int vNum, double **H, double **next_h);
double **get_next_H_matrix(int k, int vNum, double **norm_matrix, double **H);
double **read_file(const char *file_name, int vNum, int vSize);
void calc_matrix_dim(char *file_name, int *dim);
double **matrix_multiplication(double **matrix1, double **matrix2, int rows1, int cols1, int cols2);
double **calc_matrix_transpose(double **matrix, int vNum, int vSize);
void make_copy(double **targetMatrix, double **baseMatrix, int rows, int cols);
double **calc_matrix_by_goal(char *goal, double **datapoints, int vNum, int vSize);