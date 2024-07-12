double** similarityMatrix(int vNum, int vSize, double **datapoints);
double** diagonalMatrix(int vNum, int vSize, double **datapoints);
double** normalizedSimilarityMatrix(int vNum, int vSize, double **datapoints);
double** calcSymnmf(int k, int vNum, double **norm_matrix, double** H);
void printTheFormat(double** datapoints, int vNum, int vSize);
void freeFuncMem(double** matrix, int vNum);
