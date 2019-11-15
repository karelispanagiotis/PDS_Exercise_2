/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"

/* #define VERBOSE */

static char * STR_CORRECT_WRONG[] = {"WRONG", "CORRECT"};

// =================
// === UTILITIES ===
// =================

double dist(double *X, double *Y, int i, int j, int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X[l*n+i] - Y[l*m+j] ) * ( X[l*n+i] - Y[l*m+j] );
  }

  return sqrt(dist);
}


// ==================
// === VALIDATION ===
// ==================

//! kNN validator
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResult( knnresult knnres, double * corpus, double * query,
                    int n, int m, int d, int k ) {

  /* loop through all query points */
  for (int j = 0; j < m; j++ ){

    /* max distance so far (equal to kth neighbor after nested loop) */
    double maxDist = -1;
    
    /* mark all distances as not computed */
    int * visited = (int *) calloc( n, sizeof(int) );

    /* go over reported k nearest neighbors */
    for (int i = 0; i < k; i++ ){

      /* keep list of visited neighbors */
      visited[ knnres.nidx[i*m + j] ] = 1;

      /* get distance to stored index */
      double distxy = dist( corpus, query, knnres.nidx[i*m + j], j, d, n, m );

      /* make sure reported distance is correct */
      if ( abs( knnres.ndist[i*m + j] - distxy ) > 1e-8 ) return 0;
      
      /* distances should be non-decreasing */
      if ( knnres.ndist[i*m + j] < maxDist ) return 0;

      /* update max neighbor distance */
      maxDist = knnres.ndist[i*m + j];
      
    } /* for (k) -- reported nearest neighbors */

    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        /* get distance to unvisited vertex */
        double distxy = dist( corpus, query, i, j, d, n, m );
        
        /* point cannot be closer than kth distance */
        if ( distxy < maxDist ) return 0;
        
      } /* if (!visited[i]) */
      
    } /* for (i) -- unvisited notes */

    /* deallocate memory */
    free( visited );

  } /* for (j) -- query points */
  
  /* return */
  return 1;
  
}



int main()
{

  int n=897;                    // corpus
  int m=762;                    // query 
  int d=7;                      // dimensions
  int k=13;                     // # neighbors

  double  * corpus = (double * ) malloc( n*d * sizeof(double) );
  double  * query  = (double * ) malloc( m*d * sizeof(double) );

  for (int i=0;i<n*d;i++)
    corpus[i]= ( (double) (rand()%100) ) / 50;

  for (int i=0;i<m*d;i++)
    query[i]= ( (double) (rand()%100) ) / 50;

  knnresult knnres = kNN( corpus, query, n, m, d, k );

  int isValid = validateResult( knnres, corpus, query, n, m, d, k );
  
  printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValid]);

  free( corpus );
  free( query );

  return 0;
  
}
