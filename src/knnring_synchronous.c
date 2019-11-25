#include "knnring.h"
#include <mpi.h>
#include <string.h>
#include <stdlib.h>

void updateResult(knnresult* store, knnresult* new)
{
    // This function merges old and new knn results
    // since both results are sorted. The result is
    // stored inside store (1st parameter).

    int m = new->m, k = new->k; //get dimensions
    double tempDist[k*m];   //temp array, will be used in merging
    int tempId[k*m];        //temp array, will be used in merging

    memcpy(tempDist, store->ndist, k*m * sizeof(double));   //copies the data of store
    memcpy(tempId, store->nidx, k*m * sizeof(int));


    int t, n;  //indexes for temp and new arrays, used in merging  
    //for each point in query set (each column)
    for(int i=0; i<m; i++)
    {
        t = n = 0;  //all indexes point at the beginning of each array
        
        //for each of the k neighbours
        for(int j=0; j<k; j++)
        {
            //merge the arrays until k elements are complete
            if(tempDist[i + t*m] < new->ndist[i + n*m])
            {
                store->ndist[i + j*m] = tempDist[i + t*m];
                store->nidx[i + j*m] = tempId[i + t*m];
                t++;
            }
            else
            {
                store->ndist[i + j*m] = new->ndist[i + n*m];
                store->nidx[i + j*m] = new->nidx[i + n*m];
                n++;
            }
        }
    }
}

knnresult distrAllkNN(double * X, int n, int d, int k)
{
    knnresult result, tempResult;
    /* result:     Holds the updated result
     *             in each iteration
     * tempResult: Holds the kNN of local data X (query)
     *             inside received data Y (corpus)
     */   
    int numtasks, rank, next, prev, tag=1;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // determine previous and next neighbors
    prev = rank-1;
    next = rank+1;
    if (rank == 0)  prev = numtasks - 1;
    if (rank == (numtasks - 1))  next = 0;

    result = kNN(X, X, n, n, d, k); //find kNN of X inside X
    MPI_Send(X, d*n, MPI_DOUBLE, next, tag, MPI_COMM_WORLD);
    
    //Y holds the data to receive, to work with and finally send 
    double *Y = malloc(d*n *sizeof(double)); 
    for(int i=1; i < numtasks; i++)
    {
        MPI_Recv(Y, d*n, MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &status);
        
        tempResult = kNN(Y, X, n, n, d, k);
        updateResult(&result, &tempResult);

        free(tempResult.ndist);
        free(tempResult.nidx);
        
        MPI_Send(Y, d*n, MPI_DOUBLE, next, tag, MPI_COMM_WORLD);
    }

    return result;
}