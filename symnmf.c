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

double **init_matrix(int rows, int cols);
void free_matrix_memory(double **matrix, int vec_number);
double **calc_similarity_matrix(int vec_number, int vec_dim, double **d_points);
double **calc_diagonal_matrix(int vec_number, int vec_dim, double **d_points);
double **calc_normalized_similarity_matrix(int vec_number, int vec_dim, double **d_points);
double **get_next_H_matrix(int k, int vec_number, double **norm_matrix, double **H);
int has_converged(int k, int vec_number, double **H, double **next_h);
double **calc_symnmf(int k, int vec_number, double **norm_matrix, double **H);
double sum_vector_coordinates(double *v1, int vec_dim);
double calculate_squared_euclidean_distance(double *v1, double *v2, int vec_dim);
double **read_file(const char *file_name, int rows, int cols);
void calc_matrix_dim(char *file_name, int *dim);
double **matrix_multiplication(double **matrix1, double **matrix2, int rows1, int cols1, int cols2);
double **calc_matrix_transpose(double **matrix, int vec_number, int vec_dim);
double **calc_matrix_by_goal(char *goal, double **d_points, int vec_number, int vec_dim);
void make_a_copy(double **dest, double **src, int rows, int cols);
void print_matrix(double **d_points, int vec_number, int vec_dim);

/* Functions */

double **init_matrix(int rows, int cols)
{
    int i, j;
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (!matrix)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double *)calloc(cols, sizeof(double));
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

void free_matrix_memory(double **matrix, int vec_number)
{
    int i;
    if (!matrix)
    {
        return;
    }
    for (i = 0; i < vec_number; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

double **calc_similarity_matrix(int vec_number, int vec_dim, double **d_points)
{
    int i, j;
    double **matrix;
    if ((matrix = init_matrix(vec_number, vec_number)) == NULL)
    {
        return NULL;
    }
    for (i = 0; i < vec_number; i++)
    {
        for (j = 0; j < vec_number; j++)
        {
            matrix[i][j] = (i == j) ? 0 : calculate_squared_euclidean_distance(d_points[i], d_points[j], vec_dim);
        }
    }
    return matrix;
}

double **calc_diagonal_matrix(int vec_number, int vec_dim, double **d_points)
{
    int i;
    double **matrix, **sim_matrix;
    matrix = init_matrix(vec_number, vec_number);
    sim_matrix = calc_similarity_matrix(vec_number, vec_dim, d_points);
    if (matrix == NULL || sim_matrix == NULL)
    {
        free_matrix_memory(matrix, vec_number);
        free_matrix_memory(sim_matrix, vec_number);
        return NULL;
    }

    for (i = 0; i < vec_number; i++)
    {
        matrix[i][i] = sum_vector_coordinates(sim_matrix[i], vec_number);
    }
    free_matrix_memory(sim_matrix, vec_number);
    return matrix;
}

double **calc_normalized_similarity_matrix(int vec_number, int vec_dim, double **d_points)
{
    int i;
    double **res_da = NULL;
    double **final_res = NULL;
    double **A = calc_similarity_matrix(vec_number, vec_dim, d_points);
    double **D = calc_diagonal_matrix(vec_number, vec_dim, d_points);
    if (A == NULL || D == NULL)
    {
        free_matrix_memory(A, vec_number);
        free_matrix_memory(D, vec_number);
    }

    for (i = 0; i < vec_number; ++i)
    {
        D[i][i] = 1 / sqrt(D[i][i]);
    }
    res_da = matrix_multiplication(D, A, vec_number, vec_number, vec_number);
    final_res = res_da ? matrix_multiplication(res_da, D, vec_number, vec_number, vec_number) : NULL;

    free_matrix_memory(A, vec_number);
    free_matrix_memory(D, vec_number);
    free_matrix_memory(res_da, vec_number);

    return final_res;
}

double **get_next_H_matrix(int k, int vec_number, double **norm_matrix, double **H)
{
    int i, j;
    double **WH, **HH_transpose, **HH_transpose_H, **H_transpose, **next_h;
    if ((next_h = init_matrix(vec_number, k)) == NULL)
    {
        return NULL;
    }
    WH = matrix_multiplication(norm_matrix, H, vec_number, vec_number, k);
    H_transpose = calc_matrix_transpose(H, k, vec_number);
    HH_transpose = matrix_multiplication(H, H_transpose, vec_number, k, vec_number);
    HH_transpose_H = matrix_multiplication(HH_transpose, H, vec_number, vec_number, k);

    if (WH == NULL || H_transpose == NULL || HH_transpose_H == NULL || HH_transpose_H == NULL)
    {
        free_matrix_memory(next_h, vec_number);
        free_matrix_memory(H_transpose, k);
        free_matrix_memory(HH_transpose, vec_number);
        free_matrix_memory(HH_transpose_H, vec_number);
        free_matrix_memory(WH, vec_number);
        return NULL;
    }

    for (i = 0; i < vec_number; i++)
    {
        for (j = 0; j < k; j++)
        {
            double ratio = WH[i][j] / HH_transpose_H[i][j];
            next_h[i][j] = H[i][j] * (BETA * ratio + (1 - BETA));
        }
    }
    free_matrix_memory(H_transpose, k);
    free_matrix_memory(HH_transpose, vec_number);
    free_matrix_memory(HH_transpose_H, vec_number);
    free_matrix_memory(WH, vec_number);
    return next_h;
}

int has_converged(int k, int vec_number, double **H, double **next_h)
{
    int i, j;
    double norm = 0.0;

    for (i = 0; i < vec_number; i++)
    {
        for (j = 0; j < k; j++)
        {
            norm += pow((next_h[i][j] - H[i][j]), 2);
        }
    }

    return (norm < EPSILON);
}

double **calc_symnmf(int k, int vec_number, double **norm_matrix, double **H)
{
    int i;
    double **curr_h, **next_h, **temp;
    curr_h = H;
    if ((next_h = get_next_H_matrix(k, vec_number, norm_matrix, H)) == NULL)
    {
        return NULL;
    }

    for (i = 0; i < MAX_ITER && !has_converged(k, vec_number, curr_h, next_h); i++)
    {
        make_a_copy(curr_h, next_h, vec_number, k);
        temp = get_next_H_matrix(k, vec_number, norm_matrix, curr_h);
        if (!temp)
        {
            free_matrix_memory(next_h, vec_number);
            return NULL;
        }
        free_matrix_memory(next_h, vec_number);
        next_h = temp;
    }

    return next_h;
}

double sum_vector_coordinates(double *v1, int vec_dim)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < vec_dim; i++)
    {
        sum += v1[i];
    }
    return sum;
}

double calculate_squared_euclidean_distance(double *v1, double *v2, int vec_dim)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < vec_dim; i++)
    {
        sum += pow((v1[i] - v2[i]), 2);
    }
    return exp((-0.5) * sum);
}

double **read_file(const char *file_name, int rows, int cols)
{
    char *token = NULL, *line = NULL;
    double **d_points;
    int i = 0, j = 0;
    FILE *file;
    size_t line_length = 0;
    ssize_t read;

    file = fopen(file_name, "r");
    if (file == NULL)
    {
        printf("An Error Has Occoured");
        exit(EXIT_FAILURE);
    }
    if ((d_points = init_matrix(rows, cols)) == NULL)
    {
        fclose(file);
        return NULL;
    }

    while ((read = getline(&line, &line_length, file)) != -1 && i < rows)
    {
        token = strtok(line, ",");
        for (j = 0; j < cols && token != NULL; j++)
        {
            d_points[i][j] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        i++;
    }

    free(line);
    free(token);
    fclose(file);

    if (i != rows)
    {
        free_matrix_memory(d_points, rows);
        return NULL;
    }

    return d_points;
}

void calc_matrix_dim(char *file_name, int *dim)
{
    FILE *file;
    char *line = NULL;
    size_t line_length = 0;
    ssize_t read;
    int vec_number = 0, vec_dim = 1;
    char ch;

    file = fopen(file_name, "r");

    if (file == NULL)
    {
        return;
    }

    while ((ch = fgetc(file)) != '\n')
    {
        if (ch == ',')
        {
            vec_dim++;
        }
    }
    rewind(file);

    while ((read = getline(&line, &line_length, file)) != -1)
    {
        vec_number++;
    }

    dim[0] = vec_number;
    dim[1] = vec_dim;
    fclose(file);
    free(line);
}

double **matrix_multiplication(double **matrix1, double **matrix2, int rows1, int cols1, int cols2)
{
    int i, j, k;
    double **resMatrix;
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
double **calc_matrix_transpose(double **matrix, int vec_number, int vec_dim)
{
    int i, j;
    double **resMatrix;
    if ((resMatrix = init_matrix(vec_number, vec_dim)) == NULL)
    {
        return NULL;
    }

    for (i = 0; i < vec_number; i++)
    {
        for (j = 0; j < vec_dim; j++)
        {
            resMatrix[i][j] = matrix[j][i];
        }
    }
    return resMatrix;
}

void make_a_copy(double **dest, double **src, int rows, int cols)
{
    int i;
    for (i = 0; i < rows; i++)
    {
        memcpy(dest[i], src[i], cols * sizeof(double));
    }
}

double **calc_matrix_by_goal(char *goal, double **d_points, int vec_number, int vec_dim)
{
    if (!strcmp(goal, "sym"))
    {
        return calc_similarity_matrix(vec_number, vec_dim, d_points);
    }
    else if (!strcmp(goal, "ddg"))
    {
        return calc_diagonal_matrix(vec_number, vec_dim, d_points);
    }
    else if (!strcmp(goal, "norm"))
    {
        return calc_normalized_similarity_matrix(vec_number, vec_dim, d_points);
    }
    else
    {
        return NULL;
    }
}

void print_matrix(double **d_points, int vec_number, int vec_dim)
{
    int i, j;
    for (i = 0; i < vec_number; i++)
    {
        for (j = 0; j < vec_dim; j++)
        {
            printf("%.4f", d_points[i][j]);
            if (j != vec_dim - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int vec_number, vec_dim;
    double **d_points, **res_matrix;
    char *goal = argv[1];
    char *file_name = argv[2];
    int dim[2];

    if (argc != 3)
    {
        return EXIT_FAILURE;
    }

    calc_matrix_dim(file_name, dim);
    vec_number = dim[0];
    vec_dim = dim[1];

    if ((d_points = read_file(file_name, vec_number, vec_dim)) == NULL)
    {
        printf("An Error Has Occoured");
        return EXIT_FAILURE;
    }

    res_matrix = calc_matrix_by_goal(goal, d_points, vec_number, vec_dim);
    free_matrix_memory(d_points, vec_number);

    if (res_matrix == NULL)
    {
        printf("An Error Has Occoured");
        return EXIT_FAILURE;
    }

    print_matrix(res_matrix, vec_number, vec_number);
    free_matrix_memory(res_matrix, vec_number);
    return EXIT_SUCCESS;
}