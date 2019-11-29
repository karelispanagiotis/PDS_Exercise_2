#include "knnring.h"
#include <stdlib.h>
// #include <cblas-openblas>
#include <cblas.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

double square(double x) { return x * x; }

////////////////////////////////////////////////////////////////////////////

void swapDouble(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}
void swapInt(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
void quickSort(double* distArr, int* idArr, int start, int end, int inc)
{
    if(start < end)
    {
        int store = start;
        double pivot = distArr[end*inc];
        for (int i = start; i <= end; i++)
            if (distArr[i*inc] <= pivot)
            {
                swapDouble(distArr + i*inc, distArr + store*inc);
                swapInt(idArr + i, idArr + store);
                store++;
            }
        store--;

        quickSort(distArr, idArr, start, store - 1, inc);
        quickSort(distArr, idArr, store + 1, end, inc);
    }
}

////////////////////////////////////////////////////////////////////////////

double *calculateD(double *X, double *Y, int n, int m, int d)
{
    double *D = malloc(n*m * sizeof(double));          //is the Distances array [n-by-m]
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, //RowMajor, trans(X), noTrans(Y)
                n, m, d,                                 //dimensions
                -2.0, X, n, Y, m,                        //alpha, matrix(X), ldY, matrix(X), ldX
                0.0, D, m);                              //beta, matrix(D), ldD

    double *squareSumX = malloc(n * sizeof(double));
    double *squareSumY = malloc(m * sizeof(double));

    for (int i = 0; i < n; i++)
        squareSumX[i] = cblas_ddot(d, X + i, n, X + i, n); //dot product of i-th column of X with itself (sums of squares)

    for (int i = 0; i < m; i++)
        squareSumY[i] = cblas_ddot(d, Y + i, m, Y + i, m); //dot product of i-th column of Y with itself (sums of squares)

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            D[i*m + j] = sqrt(fabs(D[i*m + j] + squareSumX[i] + squareSumY[j]));

    free(squareSumX);
    free(squareSumY);

    return D;
}

////////////////////////////////////////////////////////////////////////////

knnresult kNNpartition(double *X, double *Y, int n, int m, int d, int k, int idOffset)
{
    double *D = calculateD(X, Y, n, m, d);

    knnresult result;
    result.nidx = (int *) malloc(k * m * sizeof(int));

    int *idArr = (int *) malloc(n * sizeof(int)); //is used as a temporary array
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
            idArr[j] = idOffset + j;

        quickSort(D + i, idArr, 0, n - 1, m);   //sorts each column
        // now every column, from [0..k-1] holds the k nearest neighbors
        // and idArr from [0...k-1] holds the k nearest IDs

        for(int j=0; j<k; j++)
            result.nidx[m*j + i] = idArr[j];    //copies IDs
    }
    free(idArr);
    
    //only keep the first k rows of D and free the rest of memory
    result.ndist = (double *) realloc(D, k * m * sizeof(double)); 
    result.k = k;
    result.m = m;

    return result;
}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k)
{
    return kNNpartition(X, Y, n, m, d, k, 0); //IDs start from 0
}
