   #include "mpi.h"
   #include <stdio.h>
   #include <stdlib.h>

int main(int argc, char *argv[])  {
    int numtasks, rank, next, prev, buf, tag1=1;
    MPI_Status status;   // required variable for Waitall routine

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 5;
    int *X = (int *) malloc(n * sizeof(int));
    for(int i=0; i<n; ++i){
        X[i] = rank;
    }
    int *Y = (int *) malloc(n * sizeof(int));

    // determine left and right neighbors
    prev = rank-1;
    next = rank+1;
    if (rank == 0)  prev = numtasks - 1;
    if (rank == (numtasks - 1))  next = 0;

    for(int i=0; i<numtasks; ++i){
        MPI_Send(X, n, MPI_INT, next, tag1, MPI_COMM_WORLD);

        MPI_Recv(Y, n, MPI_INT, prev, tag1, MPI_COMM_WORLD, &status);

        printf("In proccess %d, received %d from %d\n", rank, Y[4], prev);
        X = Y;
    }

    MPI_Finalize();
}