#include "knnring.h"
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

void printArr(double *arr, int n, int d) //prints an n x d array
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
            printf("%.4lf ", arr[i * d + j]);
        printf("\n");
    }
}

void printArrInt(int *arr, int n, int d) //prints an n x d array
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
            printf("%d ", arr[i * d + j]);
        printf("\n");
    }
}

double square(double x) { return x * x; }

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
void quickSort(double* distArr, int* idArr, int start, int end)
{
    if(start < end)
    {
        int store = start;
        double pivot = distArr[end];
        for (int i = start; i <= end; i++)
            if (distArr[i] <= pivot)
            {
                swapDouble(distArr + i, distArr + store);
                swapInt(idArr + i, idArr + store);
                store++;
            }
        store--;

        quickSort(distArr, idArr, start, store - 1);
        quickSort(distArr, idArr, store + 1, end);
    }
}
void quickSelect(int kpos, double *distArr, int *idArr, int start, int end)
{
    int store = start;
    double pivot = distArr[end];
    for (int i = start; i <= end; i++)
        if (distArr[i] <= pivot)
        {
            swapDouble(distArr + i, distArr + store);
            swapInt(idArr + i, idArr + store);
            store++;
        }
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, distArr, idArr, store + 1, end);
    else quickSelect(kpos, distArr, idArr, start, store - 1);
}

double *calculateD(double *X, double *Y, int n, int m, int d)
{
    double *D = malloc(m * n * sizeof(double));          //is the Distances array [m-by-n]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, //RowMajor, noTrans(X), trans(Y)
                m, n, d,                                 //dimensions
                -2.0, Y, d, X, d,                        //alpha, matrix(Y), ldY, matrix(X), ldX
                0.0, D, n);                              //beta, matrix(D), ldD

    double *squareSumX = malloc(n * sizeof(double));
    double *squareSumY = malloc(m * sizeof(double));

    for (int i = 0; i < n; i++)
        squareSumX[i] = cblas_ddot(d, &X[i*d], 1, &X[i*d], 1); //dot product of i-th row of X with itself (sums of squares)

    for (int i = 0; i < m; i++)
        squareSumY[i] = cblas_ddot(d, &Y[i*d], 1, &Y[i*d], 1); //dot product of i-th row of Y with itself (sums of squares)

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            D[i * n + j] = sqrt(D[i*n + j] + squareSumX[j] + squareSumY[i]);

    free(squareSumX);
    free(squareSumY);

    return D;
}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k)
{
    double *D = calculateD(X, Y, n, m, d);
    int *idArr = malloc(n*sizeof(int));
    knnresult result;

    result.k = k;
    result.m = m;
    result.ndist = malloc(m*k * sizeof(double));
    result.nidx = malloc(m*k * sizeof(int));

    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
            idArr[j] = j;

        //quickSelect(k-1,D + i*n, idArr, 0, n-1);
        quickSort(D + i*n, idArr, 0, n - 1);
        //now every row from [0..k-1] holds the k nearest neighbors

        for(int a=0; a<k; a++)
            result.nidx[a*m + i] = idArr[a];
        //memcpy(result.ndist + i*k, D + i*n, k*sizeof(double));  //copies the  kNN's distances
        //memcpy(result.nidx + i*k, idArr, k*sizeof(int));  //copies the kNN's IDs
    }
    for(int a=0; a<k; a++)
        for(int b=0; b<m; b++)
            result.ndist[a*m + b] = D[b*n + a];

    printArr(result.ndist,k,m);
    printArrInt(result.nidx,k,m);


    free(D);
    free(idArr);
    return result;

}
