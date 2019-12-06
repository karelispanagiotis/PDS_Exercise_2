#include "knnring.h"
// #include <mpi.h>
#include "mpi.h"
#include <string.h>
#include <stdlib.h>

void swapPtr(double** ptr1, double** ptr2)
{
    double *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

void updateResult(knnresult* store, knnresult* new)
{
    // This function merges old and new knn results
    // since both results are sorted. The result is
    // stored inside store (1st parameter).

    int m = new->m, k = new->k; //get dimensions
    double tempDist[k*m];   //temp array, will be used in merging
    int tempId[k*m];        //temp array, will be used in merging

    memcpy(tempDist, store->ndist, k * m * sizeof(double));   //copies the data of store
    memcpy(tempId, store->nidx, k * m * sizeof(int));


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

double find_max(double *A,  int size){
    double max = 0;    // Distances are positive so 0 is a good initializer for max
    for(int i=0; i<size-1; ++i){
        if(A[i] > max){
            max = A[i];
        }
    }
    return max;
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
    MPI_Request requests[2];
    MPI_Status statuses[2];

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Max calculation
    double max_proccess_temp = 0;  // max distance for this proccess with a specific query Y
    double max_proccess_total = 0; // the maximum of the above max_process_temp for all querys Y

    // determine previous and next neighbors
    prev = rank-1;
    next = rank+1;
    if (rank == 0)  prev = numtasks - 1;
    if (rank == (numtasks - 1))  next = 0;

    int idOffset = (rank-1)*n;
    if(rank == 0) idOffset = (numtasks-1)*n;
    
    double *Y = (double *) malloc(d*n*sizeof(double));     //will process and send data
    double *Z = (double *) malloc(d*n*sizeof(double));     //will receive data while processing

    MPI_Isend(X, d*n, MPI_DOUBLE, next, tag, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(Y, d*n, MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &requests[1]);
    
    result = kNNpartition(X, X, n, n, d, k, idOffset);    //IDs start from rank*n

    MPI_Waitall(2, requests, statuses);

    for(int iter=1; iter<numtasks; iter++)
    {
        MPI_Isend(Y, d*n, MPI_DOUBLE, next, tag, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(Z, d*n, MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &requests[1]);

        idOffset = ((idOffset - n) < 0) ? (idOffset - n)%(numtasks*n) + numtasks*n : (idOffset - n)%(numtasks*n); //(idOffset - 1) modulo numtasks*n
        tempResult = kNNpartition(Y, X, n, n, d, k, idOffset);
        updateResult(&result, &tempResult);


        free(tempResult.ndist);
        free(tempResult.nidx);
        
        MPI_Waitall(2, requests, statuses);
        swapPtr(&Y, &Z);
    }

    max_proccess_temp = find_max(result.ndist, result.k * result.m);
    if(max_proccess_temp > max_proccess_total) max_proccess_total = max_proccess_temp;

    // Send the max result to the proccess with id=0 and
    // compute the global maximum there
    if(rank != 0){
        MPI_Send(&max_proccess_total, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }else if(rank == 0){
        double max_p[numtasks];
        double max_global;
        for(int i=1; i<numtasks; ++i){
            MPI_Recv(&max_p[i], 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
        }
        max_p[0] = max_proccess_temp;
        max_global = find_max(max_p, numtasks); // this is the global maximum for al lpoints
        int min_global = 0; // Global minimum is 0 by default
    }

    free(Y);
    free(Z);
    return result;
}