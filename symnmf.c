#define _GNU_SOURCE
#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "symnmf.h"

#define MAX_ITER 300
#define EPSILON 0.0001
#define BETA 0.5

double** init_matrix(int rows,int cols)
{
    int i, j;
    double **matrix = (double**) malloc(rows * sizeof(double *));
    if (!matrix)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double*) calloc(cols, sizeof(double));
        if (!matrix[i])
        {
            for (j = 0; j < i; j++)
            {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void free_matrix_memory(double **matrix, int vNum)
{
    int i;
    if (!matrix)
    {
        return;
    }
    for (i = 0; i < vNum; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

void print_matrix(double** datapoints, int vNum, int vSize) {
    int i, j;
    for (i = 0; i < vNum; i++) {
        for (j=0; j < vSize; j++) {
            printf("%.4f", datapoints[i][j]);
            if (j != vSize-1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

double** calc_similarity_matrix(int vNum, int vSize, double **datapoints) {
    int i, j;
    double** matrix;
    if ((matrix = init_matrix(vNum, vNum)) == NULL) {
        return NULL;
    }
    for (i = 0; i < vNum; i++) {
        for (j = 0; j < vNum; j++) {
            matrix[i][j] = (i == j) ? 0 : calculate_squared_euclidean_distance(datapoints[i], datapoints[j], vSize);
        }
    }
    return matrix;
}

double** calc_diagonal_matrix(int vNum, int vSize, double **datapoints) {
    int i;
    double **matrix, **sim_matrix;
    matrix = init_matrix(vNum, vNum);
    sim_matrix = calc_similarity_matrix(vNum, vSize, datapoints); 
    if (matrix == NULL || sim_matrix == NULL) {
        free_matrix_memory(matrix, vNum);
        free_matrix_memory(sim_matrix, vNum);
        return NULL;
    }

    for (i = 0; i < vNum; i++) {
        matrix[i][i] = sum_vector_coordinates(sim_matrix[i], vNum);
    }
    free_matrix_memory(sim_matrix, vNum);
    return matrix;
}

double** calc_normalized_similarity_matrix(int vNum, int vSize, double **datapoints) {
    int i;
    double** res_da = NULL;
    double** final_res = NULL;
    double **A = calc_similarity_matrix( vNum, vSize, datapoints);
    double **D = calc_diagonal_matrix(vNum, vSize, datapoints);
    if (A == NULL || D == NULL) {
        free_matrix_memory(A, vNum);
        free_matrix_memory(D, vNum);
    }

    for (i = 0; i < vNum; ++i) {
        D[i][i] = 1/sqrt(D[i][i]);
    }
    res_da = multiply_matrices(D, A, vNum, vNum, vNum);
    final_res = res_da ? multiply_matrices(res_da, D, vNum, vNum, vNum) : NULL;

    free_matrix_memory(A, vNum);
    free_matrix_memory(D, vNum);
    free_matrix_memory(res_da, vNum);

    return final_res;
}

double** get_next_H_matrix(int k, int vNum, double **norm_matrix, double** H) {
    int i, j;
    double **WH, **HH_transpose, **HH_transpose_H, **H_transpose, **next_h;
    if ((next_h = init_matrix(vNum, k)) == NULL) {
        return NULL;
    }
    WH = multiply_matrices(norm_matrix, H, vNum, vNum, k);
    H_transpose = calc_matrix_transpose(H, k, vNum);
    HH_transpose = multiply_matrices(H, H_transpose, vNum, k, vNum);
    HH_transpose_H =  multiply_matrices(HH_transpose, H, vNum, vNum, k);

    if (WH == NULL || H_transpose == NULL || HH_transpose_H == NULL || HH_transpose_H == NULL)
    {
        free_matrix_memory(next_h, vNum);
        free_matrix_memory(H_transpose, k);
        free_matrix_memory(HH_transpose, vNum);
        free_matrix_memory(HH_transpose_H, vNum);
        free_matrix_memory(WH, vNum);
        return NULL;
    }

    for (i = 0; i < vNum; i++)
    {
        for (j = 0; j < k; j++)
        {
            double ratio = WH[i][j] / HH_transpose_H[i][j];
            next_h[i][j] = H[i][j] * (BETA * ratio + (1 - BETA));
        }
    }
    free_matrix_memory(H_transpose, k);
    free_matrix_memory(HH_transpose, vNum);
    free_matrix_memory(HH_transpose_H, vNum);
    free_matrix_memory(WH, vNum);
    return next_h;
}

int has_converged(int k, int vNum, double **H, double** next_h) {
    int i, j;
    double norm = 0.0;

    for (i = 0; i < vNum; i++) {
        for (j = 0; j < k; j++) {
            norm += pow((next_h[i][j]-H[i][j]), 2);
        }
    }

    return (norm < EPSILON);
}

double** calc_symnmf(int k, int vNum, double **norm_matrix, double** H) {
    int i;
    double **curr_h, **next_h, **temp;
    curr_h = H;
    if ((next_h = get_next_H_matrix(k, vNum, norm_matrix, H)) == NULL) {
        return NULL;
    }

    for (i = 0; i < MAX_ITER && !has_converged(k, vNum, curr_h, next_h); i++)
    {
        copy_matrix(curr_h, next_h, vNum, k);
        temp = get_next_H_matrix(k, vNum, norm_matrix, curr_h);
        if (!temp)
        {
            free_matrix_memory(next_h, vNum);
            return NULL;
        }
        free_matrix_memory(next_h, vNum);
        next_h = temp;
    }

    return next_h;
}

double sum_vector_coordinates(double *v1, int vSize) {
    int i;
    double sum = 0.0;
    for (i = 0; i < vSize ; i++) {
        sum += v1[i];
    }
    return sum;
}

double calculate_squared_euclidean_distance(double *v1, double *v2, int vSize) {
    int i;
    double sum = 0.0;
    for (i = 0; i < vSize ; i++) {
        sum += pow((v1[i]-v2[i]), 2);
    }
    return exp((-0.5) * sum);
}

double **read_file(const char *file_name, int rows, int cols)
{
    char *token = NULL, *line = NULL;
    double **datapoints;
    int i = 0, j = 0;
    FILE *file;
    size_t line_length = 0;
    ssize_t read;

    file = fopen(file_name, "r");
    if (file == NULL) {
        printf("An Error Has Occoured");
        exit(EXIT_FAILURE);
    }
    if ((datapoints = init_matrix(rows, cols)) == NULL)
    {
        fclose(file);
        return NULL;
    }

    while ((read = getline(&line, &line_length, file)) != -1 && i < rows) {
        token = strtok(line, ",");
        for (j = 0; j < cols && token != NULL; j++)
        {
            datapoints[i][j] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        i++;
    }

    free(line);
    free(token);
    fclose(file);

    if (i != rows)
    {
        free_matrix_memory(datapoints, rows);
        return NULL;
    }

    return datapoints;
}

void get_matrix_dim(char* file_name, int* dim) {
    FILE *file;
    char *line = NULL;
    size_t line_length = 0;
    ssize_t read;
    int vNum = 0, vSize = 1;
    char ch;

    file = fopen(file_name, "r");

    if (file == NULL) {
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

double** multiply_matrices(double** matrix1, double** matrix2, int rows1, int cols1, int cols2) {
    int i, j, k;
    double** resMatrix;
    if (!matrix1 || !matrix2 || (resMatrix = init_matrix(rows1, cols2)) == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows1; i++)
    {
        for (j = 0; j < cols2; j++)
        {
            for (k = 0; k < cols1; k++)
            {
                resMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return resMatrix;
}

double** calc_matrix_transpose(double** matrix, int vNum, int vSize) {
    int i, j;
    double **resMatrix;
    if ((resMatrix = init_matrix(vNum, vSize)) == NULL) {
        return NULL;
    }

    for (i=0; i< vNum; i++) {
        for (j=0; j< vSize; j++) {
            resMatrix[i][j] = matrix[j][i];
        }
    }
    return resMatrix;
}

void copy_matrix(double **dest, double **src, int rows, int cols)
{
    int i;
    for (i = 0; i < rows; i++)
    {
        memcpy(dest[i], src[i], cols * sizeof(double));
    }
}

double **calc_matrix_by_goal(char *goal, double **datapoints, int vNum, int vSize)
{
    if (!strcmp(goal, "sym"))
    {
        return calc_similarity_matrix(vNum, vSize, datapoints);
    }
    else if (!strcmp(goal, "ddg"))
    {
        return calc_diagonal_matrix(vNum, vSize, datapoints);
    }
    else if (!strcmp(goal, "norm"))
    {
        return calc_normalized_similarity_matrix(vNum, vSize, datapoints);
    }
    else
    {
        return NULL;
    }
}

int main(int argc, char* argv[]) {
    int vNum, vSize;
    double **datapoints, **res_matrix;
    char *goal = argv[1];
    char *file_name = argv[2];
    int dim[2];

    if (argc != 3){
        return EXIT_FAILURE;
    }

    get_matrix_dim(file_name, dim);
    vNum = dim[0];
    vSize = dim[1];

    if ((datapoints = read_file(file_name, vNum, vSize)) == NULL) {
        printf("An Error Has Occoured");
        return EXIT_FAILURE;
    }

    res_matrix = calc_matrix_by_goal(goal, datapoints, vNum, vSize);
    free_matrix_memory(datapoints, vNum);

    if (res_matrix == NULL) {
        printf("An Error Has Occoured");
        return EXIT_FAILURE;
    }
    
    print_matrix(res_matrix, vNum, vNum);
    free_matrix_memory(res_matrix, vNum);
    return EXIT_SUCCESS;
}