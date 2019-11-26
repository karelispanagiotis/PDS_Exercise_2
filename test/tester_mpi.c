/*!
  \file   tester.c
  \brief  Validate kNN ring implementation (MPI).

  \author Dimitris Floros
  \date   2019-11-25
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"
#include "mpi.h"

#include "tester_helper.h"


int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);       // initialize MPI

  int p, id;                    // # processess and PID
  int n=1423;                    // # corpus elements per process
  int d=37;                      // dimensions
  int k=13;                     // # neighbors

  double * corpus;              // will hold data
  
  MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
  MPI_Comm_size(MPI_COMM_WORLD, &p); // # tasks

  
  // initialize (as if it could hold, for testing)
  double * corpusAll = (double * ) malloc( p*n*d * sizeof(double) );

  if (id == 0) {                // ..... MASTER
    
    for (int ip = 0; ip < p; ip++){

      // "read" new chunk
      corpus = (double * ) malloc( n*d * sizeof(double) );

      for (int i=0;i<n*d;i++){
        corpusAll[i+ip*n*d] = ( (double) (rand()) ) / (double) RAND_MAX;
        corpus[i]= corpusAll[i+ip*n*d];
      }
      
      if (ip == p-1)            // last chunk is mine
        break;
      
      // which process to send? what tag?
      int dst = ip+1;
      int tag = 1;

      // send to correct process
      MPI_Send(corpus, n*d, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
      
      free( corpus );
    
    }

  } else {                      // ..... other processes

    // from which process to I receive (master)? what tag?
    int rcv = 0;
    int tag = 1;
    MPI_Status Stat;

    corpus = (double * ) malloc( n*d * sizeof(double) );
    MPI_Recv(corpus, n*d, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);
    
    
  }

  
  // run the distributed kNN code
  knnresult knnres = distrAllkNN( corpus, n, d, k);


  // ~~~~~~~~~~~~~~~~~~~~ gather back kNN results

  if (id == 0) {                // ..... MASTER

    knnresult knnresall;
    knnresall.nidx  = (int *)   malloc( n*p*k*sizeof(int)    );
    knnresall.ndist = (double *)malloc( n*p*k*sizeof(double) );
    knnresall.m = n*p;
    knnresall.k = k;

    MPI_Status Stat;
    
    for (int ip = 0; ip < p-1; ip++){

      // from which process to I receive? what tag?
      int rcv = ip+1;
      int tag = 1;
      MPI_Status Stat;
      
      MPI_Recv( &knnresall.nidx[ip*n*k], n*k, MPI_INT, rcv, tag,
                MPI_COMM_WORLD, &Stat);

      MPI_Recv( &knnresall.ndist[ip*n*k], n*k, MPI_DOUBLE, rcv, tag,
                MPI_COMM_WORLD, &Stat);

    }

    // move my result to final struct
    for (int i = 0; i < n*k; i++){
      knnresall.nidx[(p-1)*n*k+i] = knnres.nidx[i];
      knnresall.ndist[(p-1)*n*k+i] = knnres.ndist[i];
    }
    
    // ---------- validate result
    int isValidC = validateResultColMajor( knnresall, corpusAll,
                                           corpusAll,
                                           n*p, n*p, d, k );

    int isValidR = validateResultRowMajor( knnresall, corpusAll,
                                           corpusAll,
                                           n*p, n*p, d, k );

    printf("Tester validation: %s NEIGHBORS\n",
           STR_CORRECT_WRONG[isValidC||isValidR]);


  } else {                      // ..... other processes

      
      // which process to send? what tag?
      int dst = 0;
      int tag = 1;

      // send to correct process
      MPI_Send(knnres.nidx, n*k, MPI_INT, dst, tag, MPI_COMM_WORLD);
      MPI_Send(knnres.ndist, n*k, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
    
  }

  free( corpus );
  free( corpusAll );
  
  MPI_Finalize();               // clean-up
  
  return 0;
  
}
