#ifndef TESTER_HELPER_H
#define TESTER_HELPER_H

/* #define VERBOSE */

static char * STR_CORRECT_WRONG[] = {"WRONG", "CORRECT"};

// =================
// === UTILITIES ===
// =================

double distColMajor(double *X, double *Y,
                    int i, int j,
                    int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X[l*n+i] - Y[l*m+j] ) * ( X[l*n+i] - Y[l*m+j] );
  }

  return sqrt(dist);
}

double distRowMajor(double *X, double *Y,
                    int i, int j,
                    int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X[l+i*d] - Y[l+j*d] ) * ( X[l+i*d] - Y[l+j*d] );
  }

  return sqrt(dist);
}



// ==================
// === VALIDATION ===
// ==================

//! kNN validator (col major)
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResultColMajor( knnresult knnres,
                            double * corpus, double * query,
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
      double distxy = distColMajor( corpus, query, knnres.nidx[i*m + j], j, d, n, m );

      /* make sure reported distance is correct */
      if ( fabs( knnres.ndist[i*m + j] - distxy ) > 1e-8 ) return 0;
      
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
        double distxy = distColMajor( corpus, query, i, j, d, n, m );
        
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


//! kNN validator (row major)
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResultRowMajor( knnresult knnres,
                            double * corpus, double * query,
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
      visited[ knnres.nidx[i + j*k] ] = 1;

      /* get distance to stored index */
      double distxy = distRowMajor( corpus, query, knnres.nidx[i + j*k], j, d, n, m );

      /* make sure reported distance is correct */
      if ( fabs( knnres.ndist[i + j*k] - distxy ) > 1e-8 ) return 0;
      
      /* distances should be non-decreasing */
      if ( knnres.ndist[i + j*k] < maxDist ) return 0;

      /* update max neighbor distance */
      maxDist = knnres.ndist[i + j*k];
      
    } /* for (k) -- reported nearest neighbors */

    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        /* get distance to unvisited vertex */
        double distxy = distRowMajor( corpus, query, i, j, d, n, m );
        
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





#endif
