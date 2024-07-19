#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "symnmf.h"

double **parseMatrix(PyObject *X, int vNum, int vSize)
{
    double **vectors = (double **)malloc((vNum * sizeof(double *)));
    if (vectors == NULL)
    {
        fprintf(stderr, "An Error Has Occurred\n");
        exit(EXIT_FAILURE);
        int i, j;
    }

    for (i = 0; i < vNum; ++i)
    {
        vectors[i] = (double *)malloc(vSize * sizeof(double));
        if (vectors[i] == NULL)
        {
            fprintf(stderr, "An Error Has Occurred\n");
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < vSize; ++j)
        {
            vectors[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(X, i), j));
        }
    }
    return vectors;
}

PyObject *buildMatrixForPython(double **matrix, int rows, int cols)
{
    PyObject *res_matrix = PyList_New(rows);
    int i, j;
    for (i = 0; i < rows; ++i)
    {
        PyObject *vector = PyList_New(cols);
        for (j = 0; j < cols; ++j)
        {
            PyList_SetItem(vector, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(res_matrix, i, vector);
    }
    return res_matrix;
}

static PyObject *sym(PyObject *self, PyObject *args)
{
    int vNum, vSize;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiO", &vNum, &vSize, &X))
    {
        return NULL;
    }

    double **vectors = parseMatrix(X, vNum, vSize);

    /* calc final centroids using kmeans class's method */
    double **sym_matrix = similarityMatrix(vNum, vSize, vectors);
    printTheFormat(sym_matrix, vNum, vNum);
    freeFuncMem(sym_matrix, vNum);

    return Py_BuildValue("");
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    int vNum, vSize;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiO", &vNum, &vSize, &X))
    {
        return NULL;
    }

    double **vectors = parseMatrix(X, vNum, vSize);
    double **ddg_matrix = diagonalMatrix(vNum, vSize, vectors);
    printTheFormat(ddg_matrix, vNum, vNum);
    freeFuncMem(ddg_matrix, vNum);

    return Py_BuildValue("");
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    int vNum, vSize, needToPrint;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiiO", &needToPrint, &vNum, &vSize, &X))
    {
        return NULL;
    }

    double **vectors = parseMatrix(X, vNum, vSize);
    double **norm_matrix = normalizedSimilarityMatrix(vNum, vSize, vectors);
    PyObject *norm_python_matrix = buildMatrixForPython(norm_matrix, vNum, vNum);
    if (needToPrint)
    {
        printTheFormat(norm_matrix, vNum, vNum);
    }
    freeFuncMem(norm_matrix, vNum);
    return Py_BuildValue("O", norm_python_matrix);
}

static PyObject *symnmf(PyObject *self, PyObject *args)
{
    int vNum, k, analysis;
    PyObject *H;
    PyObject *W;

    if (!PyArg_ParseTuple(args, "iiOOi", &k, &vNum, &W, &H, &analysis))
    {
        return NULL;
    }

    double **H_matrix = parseMatrix(H, vNum, k);
    double **norm_matrix = parseMatrix(W, vNum, vNum);
    double **symnmf_matrix = calcSymnmf(k, vNum, norm_matrix, H_matrix);
    if (!analysis)
    {
        printTheFormat(symnmf_matrix, vNum, k);
        freeFuncMem(symnmf_matrix, vNum);
        return Py_BuildValue("");
    }
    else
    {
        PyObject *analyse_matrix = buildMatrixForPython(symnmf_matrix, vNum, k);
        freeFuncMem(symnmf_matrix, vNum);
        return Py_BuildValue("O", analyse_matrix);
    }
}

static PyMethodDef symnmfMethods[] = {
    {"sym", (PyCFunction)sym, METH_VARARGS, PyDoc_STR("sym algo")},
    {"ddg", (PyCFunction)ddg, METH_VARARGS, PyDoc_STR("ddg algo")},
    {"norm", (PyCFunction)norm, METH_VARARGS, PyDoc_STR("norm algo")},
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS, PyDoc_STR("full algo")},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef moduleDef = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    "A Python module for few algo",
    -1,
    symnmfMethods};

PyMODINIT_FUNC PyInit_mysymnmf(void)
{
    return PyModule_Create(&moduleDef);
}
