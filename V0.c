#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "supplementary.c"
#include "supplementary.h"
#include <cblas.h>

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;



struct knnresult kNN(double * X, double * Y, int n, int m, int d, int k);


//! knnresult kNN
//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
struct knnresult kNN(double * X, double * Y, int n, int m, int d, int k){
    struct knnresult result;
    result.nidx = (int*)malloc(m*k*sizeof(int));
    if(result.nidx == NULL)
        exit(1);
    result.ndist = (double*)malloc(m*k*sizeof(double));
    if(result.ndist == NULL)
        exit(1);
    result.m = m;
    result.k = k;

    //Blocking query Y to <blocks_count> blocks
    int query_chunk = 1000;
    int blocks_count = n / query_chunk;
    int *block_size = (int *)malloc(blocks_count*sizeof(int));
    for(int block_id=0; block_id<blocks_count; block_id++)
        block_size[block_id] = query_chunk*d;

    //Evenly distribute remaining query points
    int remaining_points = n - query_chunk * blocks_count;
    while(remaining_points != 0){
        for(int block_id=0; block_id<blocks_count; block_id++){
            block_size[block_id] += d;
            remaining_points --;
            if(remaining_points == 0){
                break;
            }
        }
    }

    //offset so that every blocked-query point will have its original index
    //when added to the knnresult struct
    int *index_offset = (int *)calloc(blocks_count, sizeof(int));
    if(index_offset == NULL){
        exit(1);
    }
    for(int block_id=1; block_id<blocks_count; block_id++){
        index_offset[block_id] = index_offset[block_id-1] + block_size[block_id-1];
    }

    double *X_hadamard;
    X_hadamard = (double *)malloc(n*d*sizeof(double));
    if(X_hadamard == NULL){
        exit(1);
    }
    for(int corpus_iter=0; corpus_iter<(n*d); corpus_iter++){
        X_hadamard[corpus_iter] = X[corpus_iter] * X[corpus_iter]; // X "hadamard" X
    }

    //blocking and calculating neighbors for each sub-query
    for(int block_id=0; block_id<blocks_count; block_id++){
        int block_points = block_size[block_id] / d;

        //variables to point at the real indexes of the query
        int block_start, block_end;
        block_start = index_offset[block_id];
        block_end = block_start + block_points*d ;

        double *Y_block = (double *)malloc(block_points*d*sizeof(double));
        if(Y_block == NULL){
            exit(1);
        }
        for(int query_index=block_start; query_index<block_end; query_index++){
            Y_block[query_index - block_start] = Y[query_index];
        }

        //D: Distance Matrix for each sub-query
        double *D = (double *)malloc(n*block_points*sizeof(double));
        if(D == NULL)
            exit(1);

        double *e = (double *)malloc(d*block_points*sizeof(double));
        if(e == NULL){
            exit(1);
        }
        for (int i=0; i<(d*block_points); i++){
            e[i] = 1.0;
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, block_points, d,
                    1.0, X_hadamard, d, e, d, 0.0, D, block_points);
        free(e);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, block_points, d,
                    -2.0, X, d, Y_block, d, 1.0, D, block_points);

        e = (double *)malloc(n*d*sizeof(double));
        if(e == NULL){
            exit(1);
        }
        for (int i=0; i<(n*d); i++){
            e[i] = 1.0;
        }
        for (int query_iter=0; query_iter<(block_points*d); query_iter++){
            Y_block[query_iter] = Y_block[query_iter] * Y_block[query_iter];  // Y = Y @ Y
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, block_points, d,
                    1.0, e, d, Y_block, d, 1.0, D, block_points);
        free(Y_block);
        free(e);

        for(int distances_iter=0; distances_iter<(n*block_points); distances_iter++){
            //float_absolute is used because cblas may produce a -0.00 value
            D[distances_iter] = sqrt(fabs(D[distances_iter]));
        }

        //find knn for each sub-query element
        //every column of D contains each sub-query element's distances from all corpus points
        for(int query_elem=(block_start/d); query_elem<(block_end/d); query_elem++){
            double *knn_distances = (double *)malloc(k*sizeof(double));
            if(knn_distances == NULL){
                exit(1);
            }
            int *knn_indexes = (int *)malloc(k*sizeof(int));
            if(knn_indexes == NULL){
                exit(1);
            }
            for(int nn_iter=0; nn_iter<k; nn_iter++){
                knn_indexes[nn_iter] = -1;
                knn_distances[nn_iter] = INFINITY;
            }
            //dist_iter iterates through every corpus distance for a specific column (query point)
            for(int dist_iter=0; dist_iter<n*block_points; dist_iter+=block_points){
                if(D[dist_iter+query_elem-block_start/d] <= knn_distances[k-1]){

                    //(query_elem - (block_start)/d) brings as to the right column
                    knn_distances[k-1] = D[query_elem-(block_start/d)+dist_iter];

                    knn_indexes[k-1] = (query_elem-(block_start/d)+dist_iter)/block_points;

                    if (knn_distances[k-1] < knn_distances[k-2]){
                        mergeSort_carry_one(knn_distances,knn_indexes,0,k-1);
                    }
                }
            }
            //update knn struct
            for(int knn_iter=0; knn_iter<k; knn_iter++){
                result.nidx[(k*query_elem)+knn_iter] = knn_indexes[knn_iter];
                result.ndist[(k*query_elem)+knn_iter] = knn_distances[knn_iter];
            }
            free(knn_distances);
            free(knn_indexes);
        }
        free(D);
    }
    free(index_offset);
    free(X_hadamard);

    return result;
}


int main()
{
    int N; // entire corpus size
    int d; // dimensions

    double *corpus_set = read_from_file(&N, &d);

    //! NUMBER OF NEIGHBORS
    int k = 55;

    struct timespec tic;
    struct timespec toc;

    clock_gettime( CLOCK_MONOTONIC, &tic);
    struct knnresult V0_result = kNN(corpus_set, corpus_set, N, N, d, k);
    clock_gettime( CLOCK_MONOTONIC, &toc);

    printf("V0 Duration = %f sec\n", time_spent(tic, toc));



    /*
    for(int i=0; i<N; i++){
        for(int j=0; j<k; j++){
            printf("%f  ", V0_result.ndist[ik+j]);
        }
        printf("                  ");
        for(int j=0; j<k; j++){
            printf("%d  ", V0_result.nidx[ik+j]);
        }
        printf("\n");
    }
    */

    free(corpus_set);
    return 0;
}
