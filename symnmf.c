#define _GNU_SOURCE
#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "symnmf.h"

#define MAX_ITER 300
#define EPSILON 0.0001
#define BETA 0.5
#define USAGE_ERROR 1
#define FILE_ERROR 2
#define INVALID_GOAL 3

double similarity_measure(const double *vec1, const double *vec2, int size)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < size; i++)
    {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return exp(-0.5 * sum);
}

double vector_sum(const double *vec, int size)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < size; i++)
    {
        sum += vec[i];
    }
    return sum;
}

void free_matrix(double **matrix, int rows)
{
    int i;
    if (matrix == NULL)
        return;
    for (i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

double **read_data_file(const char *file_name, int rows, int cols)
{
    double **data;
    char *line;
    int col;
    size_t len;
    ssize_t read;
    int row;
    FILE *file;
    file = fopen(file_name, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        return NULL;
    }

    data = init_matrix(rows, cols);
    if (data == NULL)
    {
        fclose(file);
        return NULL;
    }

    line = NULL;
    row = 0;
    len = 0;

    while ((read = getline(&line, &len, file)) != -1 && row < rows)
    {
        char *token = strtok(line, ",");
        for (col = 0; col < cols && token != NULL; col++)
        {
            data[row][col] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        row++;
    }

    free(line);
    fclose(file);

    if (row != rows)
    {
        fprintf(stderr, "Error: Unexpected number of rows in file\n");
        free_matrix(data, rows);
        return NULL;
    }

    return data;
}

double **matrix_multiply(double **mat1, double **mat2, int rows1, int cols1, int cols2)
{
    int i, j, k;
    double **result = init_matrix(rows1, cols2);
    if (!result)
        return NULL;

    for (i = 0; i < rows1; i++)
    {
        for (j = 0; j < cols2; j++)
        {
            for (k = 0; k < cols1; k++)
            {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

/*Functions of Project*/

double **init_matrix(int rows, int cols)
{
    int i, j;
    double **matrix = malloc(rows * sizeof(double *));
    if (!matrix)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix[i] = calloc(cols, sizeof(double));
        if (!matrix[i])
        {
            fprintf(stderr, "Memory allocation failed\n");
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

double **create_similarity_matrix(int num_points, int dim, double **points)
{
    int i, j;
    double **sim_matrix = init_matrix(num_points, num_points);
    if (!sim_matrix)
        return NULL;

    for (i = 0; i < num_points; i++)
    {
        for (j = i + 1; j < num_points; j++)
        {
            sim_matrix[i][j] = sim_matrix[j][i] = similarity_measure(points[i], points[j], dim);
        }
    }

    return sim_matrix;
}

double **create_diagonal_matrix(int num_points, int dim, double **points)
{
    int i;
    double **diag_matrix;
    double **sim_matrix;
    diag_matrix = init_matrix(num_points, num_points);
    if (!diag_matrix)
        return NULL;

    sim_matrix = create_similarity_matrix(num_points, dim, points);
    if (!sim_matrix)
    {
        free_matrix(diag_matrix, num_points);
        return NULL;
    }

    for (i = 0; i < num_points; i++)
    {
        diag_matrix[i][i] = vector_sum(sim_matrix[i], num_points);
    }

    free_matrix(sim_matrix, num_points);
    return diag_matrix;
}

double **normalize_similarity_matrix(int num_points, int dim, double **points)
{
    int i;
    /*Create similarity and diagonal degree matrices*/
    double **A;
    double **D;
    double **DA;
    double **DAD;
    A = create_similarity_matrix(num_points, dim, points);
    D = create_diagonal_matrix(num_points, dim, points);
    if (!A || !D)
    {
        free_matrix(A, num_points);
        free_matrix(D, num_points);
        return NULL;
    }

    /*Compute D^(-1/2)*/
    for (i = 0; i < num_points; i++)
    {
        D[i][i] = 1.0 / sqrt(D[i][i]);
    }

    /*Compute normalized similarity matrix: D^(-1/2) * A * D^(-1/2)*/
    DA = matrix_multiply(D, A, num_points, num_points, num_points);
    DAD = matrix_multiply(DA, D, num_points, num_points, num_points);

    free_matrix(A, num_points);
    free_matrix(D, num_points);
    free_matrix(DA, num_points);

    return DAD;
}

double **update_H(int k, int num_points, double **norm_matrix, double **H)
{
    int i, j;
    double **next_H;
    double **WH;
    double **H_transpose;
    double **HH_transpose;
    double **HH_transpose_H;
    next_H = init_matrix(num_points, k);
    if (!next_H)
        return NULL;

    WH = matrix_multiply(norm_matrix, H, num_points, num_points, k);
    H_transpose = transpose_matrix(H, k, num_points);
    HH_transpose = matrix_multiply(H, H_transpose, num_points, k, num_points);
    HH_transpose_H = matrix_multiply(HH_transpose, H, num_points, num_points, k);

    if (!WH || !H_transpose || !HH_transpose || !HH_transpose_H)
    {
        free_matrix(next_H, num_points);
        free_matrix(WH, num_points);
        free_matrix(H_transpose, k);
        free_matrix(HH_transpose, num_points);
        free_matrix(HH_transpose_H, num_points);
        return NULL;
    }

    for (i = 0; i < num_points; i++)
    {
        for (j = 0; j < k; j++)
        {
            double ratio = WH[i][j] / HH_transpose_H[i][j];
            next_H[i][j] = H[i][j] * (BETA * ratio + (1 - BETA));
        }
    }

    free_matrix(WH, num_points);
    free_matrix(H_transpose, k);
    free_matrix(HH_transpose, num_points);
    free_matrix(HH_transpose_H, num_points);

    return next_H;
}

int has_converged(int k, int num_points, double **H, double **next_H)
{
    int i, j;
    double norm = 0.0;
    for (i = 0; i < num_points; i++)
    {
        for (j = 0; j < k; j++)
        {
            double diff = next_H[i][j] - H[i][j];
            norm += diff * diff;
        }
    }
    return (norm < EPSILON);
}

double **calculate_symnmf(int k, int num_points, double **norm_matrix, double **H)
{
    int iter;
    double **curr_h;
    double **next_H;
    double **temp;
    curr_h = H;
    next_H = update_H(k, num_points, norm_matrix, H);
    if (!next_H)
        return NULL;

    for (iter = 0; iter < MAX_ITER && !has_converged(k, num_points, curr_h, next_H); iter++)
    {
        copy_matrix(curr_h, next_H, num_points, k);
        temp = update_H(k, num_points, norm_matrix, curr_h);
        if (!temp)
        {
            free_matrix(next_H, num_points);
            return NULL;
        }
        free_matrix(next_H, num_points);
        next_H = temp;
    }

    return next_H;
}

int get_dimensions(const char *filename, int *num_points, int *num_features)
{
    FILE *file;
    char ch;
    char *line;
    size_t len;
    file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        return 0;
    }

    *num_features = 0;
    *num_points = 0;

    /*Check if file is empty*/
    if ((ch = fgetc(file)) == EOF)
    {
        fprintf(stderr, "Error: File is empty\n");
        fclose(file);
        return 0;
    }
    ungetc(ch, file);

    /*Count features in the first line*/
    while ((ch = fgetc(file)) != '\n' && ch != EOF)
    {
        if (ch == ',')
            (*num_features)++;
    }
    (*num_features)++; /*Count the last feature*/

    /* Count the number of points (lines)*/
    line = NULL;
    len = 0;
    while (getline(&line, &len, file) != -1)
    {
        (*num_points)++;
    }

    free(line);
    fclose(file);
    return 1;
}

double **transpose_matrix(double **matrix, int rows, int cols)
{
    int i, j;
    double **result;
    result = init_matrix(cols, rows);
    if (!result)
        return NULL;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

void copy_matrix(double **dest, double **src, int rows, int cols)
{
    int i;
    for (i = 0; i < rows; i++)
    {
        memcpy(dest[i], src[i], cols * sizeof(double));
    }
}

void print_matrix(double **matrix, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1)
                printf(",");
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    const char *filename;
    double **data_points;
    double **norm_matrix;
    double **H;
    double **result_matrix;
    int k, num_points, num_features;

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <filename> <k>\n", argv[0]);
        return USAGE_ERROR;
    }

    filename = argv[1];
    k = atoi(argv[2]); /*Convert k from string to integer*/
    if (!get_dimensions(filename, &num_points, &num_features))
    {
        fprintf(stderr, "Error reading file dimensions\n");
        return FILE_ERROR;
    }

    data_points = read_data_file(filename, num_points, num_features);
    if (!data_points)
    {
        fprintf(stderr, "Error reading data from file\n");
        return FILE_ERROR;
    }

    /*Perform SYMNMF algorithm*/
    norm_matrix = normalize_similarity_matrix(num_points, num_features, data_points);
    if (!norm_matrix)
    {
        fprintf(stderr, "Error computing normalized similarity matrix\n");
        free_matrix(data_points, num_points);
        return 1;
    }

    /* Initialize H matrix with zeros using the existing init_matrix function */
    H = init_matrix(num_points, k);
    if (!H)
    {
        fprintf(stderr, "Error initializing H matrix\n");
        free_matrix(data_points, num_points);
        free_matrix(norm_matrix, num_points);
        return 1;
    }

    result_matrix = calculate_symnmf(k, num_points, norm_matrix, H);
    if (!result_matrix)
    {
        fprintf(stderr, "Error computing SYMNMF\n");
        free_matrix(data_points, num_points);
        free_matrix(norm_matrix, num_points);
        free_matrix(H, num_points);
        return 1;
    }

    print_matrix(result_matrix, num_points, k);

    free_matrix(data_points, num_points);
    free_matrix(norm_matrix, num_points);
    free_matrix(H, num_points);
    free_matrix(result_matrix, num_points);

    return 0;
}