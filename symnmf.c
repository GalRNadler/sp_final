#define _GNU_SOURCE
#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "symnmf.h"

void freeFuncMem(double** matrix, int vNum);
void printTheFormat(double** datapoints, int vNum, int vSize);
double sumVector(double *v1, int vSize);
double calcSqrtDis(double *v1, double *v2, int vSize);
double** similarityMatrix(int vNum, int vSize, double **datapoints);
double** initializeMatrix(int vNum,int vSize);
double** diagonalMatrix(int vNum, int vSize, double **datapoints);
double** normalizedSimilarityMatrix(int vNum, int vSize, double **datapoints);
double** calcSymnmf(int k, int vNum, double **norm_matrix, double** H);
int isConvergence(int k, int vNum, double **H, double** next_h);
double** getNextH(int k, int vNum, double **norm_matrix, double** H);
double** readFromFile(char* file_name, int vNum, int vSize);
void getMatrixDim(char* file_name, int* dim);
double** multiplyMatrix(double** matrix1, double** matrix2, int rows1, int cols1, int cols2);
double** transposeMatrix(double** matrix, int vNum, int vSize);
void copy_matrix(double** targetMatrix, double** baseMatrix, int rows, int cols);

double EPSILON = 0.0001;
int MAX_ITER = 300;

double** initializeMatrix(int vNum,int vSize) {
    int i, j;
    double** matrix = (double **) malloc(vNum * sizeof(double *));
    if (matrix == NULL) {
        fprintf(stderr, "An Error Has Occurred\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < vNum; i++) {
        matrix[i] = (double *) malloc(vSize * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "An Error Has Occurred\n");
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < vSize; j++) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

void printTheFormat(double** datapoints, int vNum, int vSize) {
    int i, j;
    for (i=0; i<vNum; i++) {
        for (j=0; j<vSize; j++) {
            printf("%.4f", datapoints[i][j]);
            if (j!=vSize-1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

double** similarityMatrix(int vNum, int vSize, double **datapoints) {
    int i, j;
    double** matrix = initializeMatrix(vNum, vNum);
    for (i = 0; i < vNum; i++) {
        for (j = 0; j < vNum; j++) {
            if (i == j) {
                matrix[i][j] = 0;
            } else {
                matrix[i][j] = calcSqrtDis(datapoints[i], datapoints[j], vSize);
            }
        }
    }
    return matrix;
}

double** diagonalMatrix(int vNum, int vSize, double **datapoints) {
    int i, j;
    double** matrix = initializeMatrix(vNum, vNum);
    double** sym_matrix = similarityMatrix(vNum, vSize, datapoints);

    for (i = 0; i < vNum; i++) {
        for (j = 0; j < vNum; j++) {
            if (i != j) {
                matrix[i][j] = 0;
            } else {
                matrix[i][j] = sumVector(sym_matrix[i], vNum);
            }
        }
    }
    freeFuncMem(sym_matrix, vNum);
    return matrix;
}

double** normalizedSimilarityMatrix(int vNum, int vSize, double **datapoints) {
    int i;
    double** res_da;
    double** final_res;
    double **A = similarityMatrix( vNum, vSize, datapoints);
    double **D = diagonalMatrix(vNum, vSize, datapoints);
    for (i = 0; i < vNum; ++i) {
        D[i][i] = 1/ sqrt(D[i][i]);
    }
    res_da = multiplyMatrix(D, A, vNum, vNum, vNum);
    final_res = multiplyMatrix(res_da, D, vNum, vNum, vNum);
    freeFuncMem(A, vNum);
    freeFuncMem(D, vNum);
    freeFuncMem(res_da, vNum);
    return final_res;
}

double** getNextH(int k, int vNum, double **norm_matrix, double** H) {
    int i, j;
    double top, bottom, tmp;
    double beta = 0.5;
    double** w_h_matrix;
    double** h_hTranspose_matrix;
    double** total_h_multiply;
    double** trsaponse_h;
    double** next_h = initializeMatrix(vNum, k);

    w_h_matrix = multiplyMatrix(norm_matrix, H, vNum, vNum, k);
    trsaponse_h = transposeMatrix(H, k, vNum);
    h_hTranspose_matrix =  multiplyMatrix(H, trsaponse_h, vNum, k, vNum);
    total_h_multiply =  multiplyMatrix(h_hTranspose_matrix, H, vNum, vNum, k);

    for (i=0; i< vNum ;i++) {
        for (j=0; j< k; j++) {
            top = w_h_matrix[i][j];
            bottom = total_h_multiply[i][j];
            tmp = (top / bottom) * beta;
            tmp = (1- beta) + tmp;
            next_h[i][j] = H[i][j] * tmp;
        }
    }
    freeFuncMem(w_h_matrix, vNum);
    freeFuncMem(trsaponse_h, k);
    freeFuncMem(h_hTranspose_matrix, vNum);
    freeFuncMem(total_h_multiply, vNum);
    return next_h;
}

int isConvergence(int k, int vNum, double **H, double** next_h) {
    int i, j;
    double norm = 0.0;

    for (i = 0; i < vNum; i++) {
        for (j = 0; j < k; j++) {
            norm += pow((next_h[i][j]-H[i][j]), 2);
        }
    }

    if (norm < EPSILON) {
        return 1;
    }
    return 0;
}

double** calcSymnmf(int k, int vNum, double **norm_matrix, double** H) {
    int currIter = 0;
    double** curr_h;
    double** next_h;
    curr_h = H;
    next_h = getNextH(k, vNum, norm_matrix, H);
    while ((isConvergence(k, vNum, curr_h, next_h) == 0) && currIter < MAX_ITER) {
        copy_matrix(curr_h, next_h,vNum, k);
        next_h = getNextH(k, vNum, norm_matrix, H);
        currIter++;
    }

    return next_h;
}

double sumVector(double *v1, int vSize) {
    int i;
    double sum = 0.0;
    for (i = 0; i < vSize ; i++) {
        sum += v1[i];
    }
    return sum;
}

double calcSqrtDis(double *v1, double *v2, int vSize) {
    int i;
    double sum = 0.0;
    for (i = 0; i < vSize ; i++) {
        sum += pow((v1[i]-v2[i]), 2);
    }
    return exp((-0.5) * sum);
}

void freeFuncMem(double** matrix, int vNum){
    int i;
    for (i=0; i<vNum; i++){
        free(matrix[i]);
    }
    free(matrix);
}

double** readFromFile(char* file_name, int vNum, int vSize) {
    FILE *file;
    char *line = NULL;
    size_t line_length = 0;
    ssize_t read;
    double** datapoints;
    double point;
    char *token;
    int row = 0, col;

    file = fopen(file_name, "r");
    if (file == NULL) {
        fprintf(stderr, "An Error Has Occurred\n");
        exit(EXIT_FAILURE);
    }
    datapoints = initializeMatrix(vNum, vSize);

    while ((read = getline(&line, &line_length, file)) != -1) {
        col = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            point = strtod(token, NULL);
            datapoints[row][col] = point;
            col ++;
            token = strtok(NULL, ",");
        }
        row ++;
    }

    fclose(file);
    free(line);
    free(token);

    return datapoints;
}

void getMatrixDim(char* file_name, int* dim) {
    FILE *file;
    char *line = NULL;
    size_t line_length = 0;
    ssize_t read;
    int vNum = 0, vSize = 1;
    char ch;

    file = fopen(file_name, "r");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    while ((ch = fgetc(file)) != '\n') {
        if (ch == ',') {
            vSize ++;
        }
    }
    rewind(file);

    while ((read = getline(&line, &line_length, file)) != -1) {
        vNum++;
    }

    dim[0] = vNum;
    dim[1] = vSize;
    fclose(file);
    free(line);
}

double** multiplyMatrix(double** matrix1, double** matrix2, int rows1, int cols1, int cols2) {
    int i, j, k;
    double** resMatrix = initializeMatrix(rows1, cols2);

    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            for (k = 0; k < cols1; k++) {
                resMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return resMatrix;
}

double** transposeMatrix(double** matrix, int vNum, int vSize) {
    int i, j;
    double** resMatrix = initializeMatrix(vNum, vSize);

    for (i=0; i< vNum; i++) {
        for (j=0; j< vSize; j++) {
            resMatrix[i][j] = matrix[j][i];
        }
    }
    return resMatrix;
}

void copy_matrix(double** targetMatrix, double** baseMatrix, int rows, int cols) {
    int i,j;
    for(i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            targetMatrix[i][j] = baseMatrix[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    int vNum, vSize;
    double** datapoints;
    double** res_matrix;
    char* goal = argv[1];
    char* file_name = argv[2];
    int* dim = calloc(sizeof(int), 2);

    if (argc != 3){
        return 0;
    }

    getMatrixDim(file_name, dim);
    vNum = dim[0];
    vSize = dim[1];

    datapoints = readFromFile(file_name, vNum, vSize);

    if (!strcmp(goal, "sym")) {
        res_matrix = similarityMatrix(vNum, vSize, datapoints);
    }
    else if (!strcmp(goal, "ddg")) {
        res_matrix = diagonalMatrix(vNum, vSize, datapoints);
    }
    else if (!strcmp(goal, "norm")) {
        res_matrix = normalizedSimilarityMatrix(vNum, vSize, datapoints);
    }
    else {
        return 1;
    }
    freeFuncMem(datapoints, vNum);
    printTheFormat(res_matrix, vNum, vNum);
    freeFuncMem(res_matrix, vNum);
    return 0;
}