#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "symnmf.h"

static double **matrix_parse(PyObject *X, int rows, int cols)
{
    double **matrix = (double **)malloc(rows * sizeof(double *));
    int i, j, k;
    if (!matrix)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix");
        return NULL;
    }

    for (i = 0; i < rows; ++i)
    {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (!matrix[i])
        {
            for (j = 0; j < i; ++j)
            {
                free(matrix[j]);
            }
            free(matrix);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix row");
            return NULL;
        }

        PyObject *row = PyList_GetItem(X, i);
        for (j = 0; j < cols; ++j)
        {
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
            if (PyErr_Occurred())
            {
                for (k = 0; k <= i; ++k)
                {
                    free(matrix[k]);
                }
                free(matrix);
                return NULL;
            }
        }
    }
    return matrix;
}

static PyObject *build_mat_Python(double **matrix, int rows, int cols)
{
    PyObject *py_matrix = PyList_New(rows);
    int i, j;
    if (!py_matrix)
        return NULL;

    for (i = 0; i < rows; ++i)
    {
        PyObject *row = PyList_New(cols);
        if (!row)
        {
            Py_DECREF(py_matrix);
            return NULL;
        }

        for (j = 0; j < cols; ++j)
        {
            PyObject *val = PyFloat_FromDouble(matrix[i][j]);
            if (!val)
            {
                Py_DECREF(row);
                Py_DECREF(py_matrix);
                return NULL;
            }
            PyList_SET_ITEM(row, j, val);
        }
        PyList_SET_ITEM(py_matrix, i, row);
    }
    return py_matrix;
}

static PyObject *similarity_matrix(PyObject *self, PyObject *args)
{
    int vec_number, vec_dim;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiO", &vec_number, &vec_dim, &X))
    {
        return NULL;
    }

    double **vectors = matrix_parse(X, vec_number, vec_dim);
    if (!vectors)
        return NULL;

    double **sym_matrix = calc_similarity_matrix(vec_number, vec_dim, vectors);
    if (!sym_matrix)
    {
        free_matrix_memory(vectors, vec_number);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create similarity matrix");
        return NULL;
    }

    print_matrix(sym_matrix, vec_number, vec_number);
    free_matrix_memory(vectors, vec_number);
    free_matrix_memory(sym_matrix, vec_number);

    Py_RETURN_NONE;
}

static PyObject *diagonal_matrix(PyObject *self, PyObject *args)
{
    int vec_number, vec_dim;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiO", &vec_number, &vec_dim, &X))
    {
        return NULL;
    }

    double **vectors = matrix_parse(X, vec_number, vec_dim);
    if (!vectors)
        return NULL;

    double **ddg_matrix = calc_diagonal_matrix(vec_number, vec_dim, vectors);
    if (!ddg_matrix)
    {
        free_matrix_memory(vectors, vec_number);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create diagonal matrix");
        return NULL;
    }

    print_matrix(ddg_matrix, vec_number, vec_number);
    free_matrix_memory(vectors, vec_number);
    free_matrix_memory(ddg_matrix, vec_number);

    Py_RETURN_NONE;
}

static PyObject *norm_matrix(PyObject *self, PyObject *args)
{
    int vec_number, vec_dim, need_to_print;
    PyObject *X;
    double **vectors;
    double **norm_matrix;

    if (!PyArg_ParseTuple(args, "iiiO", &need_to_print, &vec_number, &vec_dim, &X))
    {
        return NULL;
    }

    **vectors = matrix_parse(X, vec_number, vec_dim);
    if (!vectors)
        return NULL;

    **norm_matrix = calc_normalized_similarity_matrix(vec_number, vec_dim, vectors);
    if (!norm_matrix)
    {
        free_matrix_memory(vectors, vec_number);
        PyErr_SetString(PyExc_RuntimeError, "Failed to normalize similarity matrix");
        return NULL;
    }

    PyObject *py_norm_matrix = NULL;
    if (need_to_print)
    {
        print_matrix(norm_matrix, vec_number, vec_number);
    }
    else
    {
        py_norm_matrix = build_mat_Python(norm_matrix, vec_number, vec_number);
    }

    free_matrix_memory(vectors, vec_number);
    free_matrix_memory(norm_matrix, vec_number);

    if (need_to_print)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return py_norm_matrix;
    }
}

static PyObject *symnmf(PyObject *self, PyObject *args)
{
    int vec_number, k, analysis;
    PyObject *H, *W;

    if (!PyArg_ParseTuple(args, "iiOOi", &k, &vec_number, &W, &H, &analysis))
    {
        return NULL;
    }

    double **H_matrix = matrix_parse(H, vec_number, k);
    if (!H_matrix)
        return NULL;

    double **norm_matrix = matrix_parse(W, vec_number, vec_number);
    if (!norm_matrix)
    {
        free_matrix_memory(H_matrix, vec_number);
        return NULL;
    }

    double **symnmf_matrix = calc_symnmf(k, vec_number, norm_matrix, H_matrix);
    if (!symnmf_matrix)
    {
        free_matrix_memory(H_matrix, vec_number);
        free_matrix_memory(norm_matrix, vec_number);
        PyErr_SetString(PyExc_RuntimeError, "Failed to calculate SYMNMF");
        return NULL;
    }

    PyObject *result = NULL;
    if (analysis)
    {
        result = build_mat_Python(symnmf_matrix, vec_number, k);
    }
    else
    {
        print_matrix(symnmf_matrix, vec_number, k);
        Py_INCREF(Py_None);
        result = Py_None;
    }

    free_matrix_memory(H_matrix, vec_number);
    free_matrix_memory(norm_matrix, vec_number);
    free_matrix_memory(symnmf_matrix, vec_number);

    return result;
}

static PyMethodDef symnmf_methods[] = {
    {"similarity_matrix", (PyCFunction)similarity_matrix, METH_VARARGS, "Compute similarity matrix"},
    {"diagonal_matrix", (PyCFunction)diagonal_matrix, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm_matrix", (PyCFunction)norm_matrix, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS, "Perform SYMNMF algorithm"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    "A Python module for SYMNMF algorithm",
    -1,
    symnmf_methods};

PyMODINIT_FUNC PyInit_mysymnmf(void)
{
    return PyModule_Create(&moduledef);
}
