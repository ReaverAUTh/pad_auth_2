#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "supplementary.h"
#include "supplementary.c"
#include <cblas.h>
#include <mpi.h>


double *calculate_distance_matrix(double *X, double *Y, int n, int m, int d);
struct knnresult update_knn(struct knnresult result, double *D, int n, int m, int k, int *empty_knn, int index_offset, int block_index_offset);

//! Global variables

// block size for each process (size: m*d)
int *block_size;

//! Global varialbes end


// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


//! Compute distributed all-kNN of points in X
/*!

  \param  X      Data points                     [n-by-d]
  \param  n      Number of data points           [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
struct knnresult distrAllkNN(double *X, int n, int d, int k){
    struct knnresult result;

    result.nidx = (int*)malloc(n*k*sizeof(int));
    if(result.nidx == NULL){
        exit(1);
	}

    result.ndist = (double*)malloc(n*k*sizeof(double));
    if(result.ndist == NULL){
        exit(1);
	}

    result.m = n;  // n = (points received from world_rank=0 that handled the scattering)
    result.k = k;

	// MPI data
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int processes_count;
    MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

	int m = result.m;
	int rcved_query_count;
	int empty_knn = 1;
    //double *D;

	// Y is used for the parallel computation
	double *Y = (double *)malloc(m*d*sizeof(double));
    if(Y == NULL){
        exit(1);
	}

	// Z is used for receiving the points of the previous process (in a ring)
	double *Z;

	// buffer for sending to the next process (in a ring)
	double *send_buff;

	// for the first computation, its process searches against its own block
    for(int i=0; i<(m*d); i++){
        Y[i] = X[i];
    }

    MPI_Status status[2];
    MPI_Request requests[2];

    //offset so that every blocked-corpus point will have its original index
    //when added to the knnresult struct
    int *index_offset = (int *)calloc(processes_count, sizeof(int));
    if(index_offset == NULL){
        exit(1);
    }
    for(int block_id=1; block_id<processes_count; block_id++){
        index_offset[block_id] = index_offset[block_id-1] + (block_size[block_id-1]/d);
    }
    for(int ring_iter=0; ring_iter<processes_count; ring_iter++){

        // the last iteration won't S/R and will only compute
        if(ring_iter != (processes_count-1)){

            //! RECEIVING
            // the first process must receive from the last one
            if (world_rank == 0){
                rcved_query_count = block_size[processes_count-1] / d;
                Z = (double *)malloc(rcved_query_count*d*sizeof(double));
                if(Z == NULL){
                    exit(1);
                }
                MPI_Irecv(Z, rcved_query_count*d, MPI_DOUBLE, processes_count-1, processes_count-1, MPI_COMM_WORLD, &requests[0]);
            }
            // everything else receives from the previous process
            else{
                rcved_query_count = block_size[world_rank-1] / d;
                Z = (double *)malloc(rcved_query_count*d*sizeof(double));
                if(Z == NULL){
                    exit(1);
                }
                MPI_Irecv(Z, rcved_query_count*d, MPI_DOUBLE, world_rank-1, world_rank-1, MPI_COMM_WORLD, &requests[0]);
            }

            //! SENDING
            send_buff = (double *)malloc(m*d*sizeof(double));
            if(send_buff == NULL){
                exit(1);
            }
            for(int points_iter=0; points_iter<(m*d); points_iter++){
                send_buff[points_iter] = Y[points_iter];
            }
            // the last process must send to the first one
            if (world_rank == (processes_count-1)){
                MPI_Isend(send_buff, m*d, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD, &requests[1]);
            }
            // everything else sends to the next process
            else{
                MPI_Isend(send_buff, m*d, MPI_DOUBLE, world_rank+1, world_rank, MPI_COMM_WORLD, &requests[1]);
            }
        }

        //! Parallel computation, while MPI is communicating with asynchronous S/R

        // The query must be further blocked, because for large n and small process_count, m will be large
        int query_chunk = 4096;
        int blocks_count;
        if (m <= query_chunk){
            blocks_count = 1;
        }
        else {
            blocks_count = m / query_chunk;
        }

        int *sub_block_size = (int *)malloc(blocks_count*sizeof(int));
        if (sub_block_size == NULL){
            exit(1);
        }
        for(int block_id=0; block_id<blocks_count; block_id++){
            if (m <= query_chunk){
                sub_block_size[block_id] = m*d;
            }
            else{
                sub_block_size[block_id] = query_chunk*d;
            }
        }
        if (m > query_chunk){
            //Evenly distribute remaining query points
            int remaining_points = m - query_chunk * blocks_count;
            while(remaining_points != 0){
                for(int block_id=0; block_id<blocks_count; block_id++){
                    sub_block_size[block_id] += d;
                    remaining_points --;
                    if(remaining_points == 0){
                        break;
                    }
                }
            }
        }

        //offset so that every sub-blocked-query point will have its original index
        //when added to the knnresult struct
        int *block_index_offset = (int *)calloc(blocks_count, sizeof(int));
        if(block_index_offset == NULL){
            exit(1);
        }
        for(int block_id=1; block_id<blocks_count; block_id++){
            block_index_offset[block_id] = block_index_offset[block_id-1] + sub_block_size[block_id-1];
        }

        for(int block_iter=0; block_iter<blocks_count; block_iter++){

            // sub_block_points == m for each calculation
            int sub_block_points = sub_block_size[block_iter]/d;

            int block_start, block_end;
            block_start = block_index_offset[block_iter];
            block_end = block_start + sub_block_points*d;

            double *Y_block = (double *)malloc(sub_block_points*d*sizeof(double));
            if(Y_block == NULL){
                exit(1);
            }
            for(int query_index=block_start; query_index<block_end; query_index++){
                Y_block[query_index - block_start] = Y[query_index];
            }

            //D: Distance Matrix for each sub-query
            double *D = (double *)malloc(n*sub_block_points*sizeof(double));
            if(D == NULL){
                exit(1);
            }
            D = calculate_distance_matrix(X, Y_block, n, sub_block_points, d);
            free(Y_block);
            result = update_knn(result, D, n, sub_block_points, k, &empty_knn, index_offset[world_rank], block_index_offset[block_iter]);
            free(D);
        }
        free(block_index_offset);
        free(sub_block_size);

        //! End of parallel computation

        if(ring_iter != processes_count-1){

            MPI_Waitall(2, requests, status);
            free(Y);
            free(send_buff);
            m = rcved_query_count;
            Y = (double *)malloc(m*d*sizeof(double));
            if(Y == NULL){
                exit(1);
            }
            for(int i=0; i<(m*d); i++){
                Y[i] = Z[i];
            }
            free(Z);

            // block_size and offsets must be updated because new
            // points were received through the MPI
            int *previous_size = (int *)malloc(processes_count*sizeof(int));
            if(previous_size == NULL){
                exit(1);
            }
            int *previous_offset = (int *)malloc(processes_count*sizeof(int));
            if(previous_offset == NULL){
                exit(1);
            }
            for(int process_iter=0; process_iter<processes_count; process_iter++){
                previous_size[process_iter] = block_size[process_iter];
                previous_offset[process_iter] = index_offset[process_iter];
            }
            for(int process_iter=0; process_iter<processes_count; process_iter++){
                if(process_iter==0){
                    block_size[process_iter] = previous_size[processes_count-1];
                    index_offset[process_iter] = previous_offset[processes_count-1];
                }
                else{
                    block_size[process_iter] = previous_size[process_iter-1];
                    index_offset[process_iter] = previous_offset[process_iter-1];
                }
            }
            free(previous_size);
            free(previous_offset);
        }
    }
    free(Y);

    return result;
}

int main(){

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int processes_count;
    MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

    double *corpus_set;
    double *X_blocked; // block that each process will receive
    int N; // entire corpus size
    int n; // size of each block (counts points)
    int d; // dimensions

    block_size = (int *)malloc(processes_count*sizeof(int));
    if (block_size == NULL){
        exit(1);
    }
    // array of size p, says where an incision will be made in the corpus
    // for every process
    int *index_offset = (int *)calloc(processes_count, sizeof(int));
    if (index_offset == NULL){
        exit(1);
    }

    //! PID #0 will handle scattering
    if(world_rank == 0){
        //! Call read_from_file
        corpus_set = read_from_file(&N, &d);

        int block = N / processes_count;
        int remaining_points = N - block * processes_count;
        for(int process_iter=0; process_iter<processes_count; process_iter++){
            block_size[process_iter] = block*d;
        }

        while(remaining_points != 0){
            for(int process_iter=0; process_iter<processes_count; process_iter++){
                block_size[process_iter] += d;
                remaining_points --;
                if(remaining_points == 0){
                    break;
                }
            }
        }
        for(int p_iter=1; p_iter<processes_count; p_iter++){
            index_offset[p_iter] = index_offset[p_iter-1] + block_size[p_iter-1];
        }
    }

    //! Broadcast everything to the rest processes
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(block_size, processes_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index_offset, processes_count, MPI_INT, 0, MPI_COMM_WORLD);

    n = block_size[world_rank] / d;

    // this works as a max size for the buffer
    X_blocked = (double *)malloc(block_size[0]*sizeof(double));
    if (X_blocked == NULL){
        exit(1);
    }

    // send every process its corpus block
    MPI_Scatterv(corpus_set, block_size, index_offset, MPI_DOUBLE, X_blocked, block_size[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //! NUMBER OF NEIGHBORS
    int k = 55;

    struct timespec tic;
    if(world_rank ==0)
          clock_gettime( CLOCK_MONOTONIC, &tic);

    struct knnresult V1_result = distrAllkNN(X_blocked,n,d,k);

    // wait for everything to get here, so that process 0 can count
    // the duration correctly and accumulate all data
    MPI_Barrier(MPI_COMM_WORLD);

    struct timespec toc;
    if(world_rank ==0){
          clock_gettime( CLOCK_MONOTONIC, &toc);
          printf("V1 Duration = %f sec\n", time_spent(tic, toc));
    }

    //! Combine all process results into one struct, that of #0
    struct knnresult final_result;
    if(world_rank == 0){

        final_result.nidx = (int *) malloc(N*k*sizeof(int));
        if(final_result.nidx == NULL){
            exit(1);
        }
        final_result.ndist = (double *) malloc(N*k*sizeof(double));
        if(final_result.ndist == NULL){
            exit(1);
        }
        final_result.k = k;
        final_result.m = N;

        // add results of process #0
        for(int i=0; i<(n*k); i++){
            final_result.nidx[i] = V1_result.nidx[i];
            final_result.ndist[i] = V1_result.ndist[i];
        }
    }

    // results buffer will receive the results from the rest of the processes
    struct knnresult results_buffer[processes_count - 1];

    for(int p_iter=1; p_iter<processes_count; p_iter++){

        // m S/R

        //everyone else is sending
        if(world_rank == p_iter){
            MPI_Send(&n, 1, MPI_INT, 0, p_iter, MPI_COMM_WORLD);
        }
        //#0 is receiving
        else if(world_rank == 0){
            MPI_Recv(&results_buffer[p_iter-1].m, 1, MPI_INT, p_iter, p_iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            results_buffer[p_iter-1].nidx = (int *)malloc(results_buffer[p_iter-1].m*k*sizeof(int));
            if(results_buffer[p_iter-1].nidx == NULL){
                exit(1);
            }
            results_buffer[p_iter-1].ndist = (double *)malloc(results_buffer[p_iter-1].m*k*sizeof(double));
            if(results_buffer[p_iter-1].ndist == NULL){
                exit(1);
            }
        }
        // ndist S/R

        if(world_rank == p_iter){
            MPI_Send(V1_result.ndist, n*k, MPI_DOUBLE, 0, p_iter, MPI_COMM_WORLD);
        }
        else if(world_rank == 0){
            MPI_Recv(results_buffer[p_iter-1].ndist, results_buffer[p_iter-1].m*k, MPI_DOUBLE, p_iter, p_iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // nidx S/R

        if(world_rank == p_iter){
            MPI_Send(V1_result.nidx, n*k, MPI_INT, 0, p_iter, MPI_COMM_WORLD);
        }
        else if(world_rank == 0){
            MPI_Recv(results_buffer[p_iter-1].nidx, results_buffer[p_iter-1].m*k, MPI_INT, p_iter, p_iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    //! Combining the results
    if(world_rank == 0){
        //results_offset will show where pid #0 must store the next results
        int results_offset = n*k;

        for(int p_iter=1; p_iter<processes_count; p_iter++){
            for(int i=0; i<results_buffer[p_iter-1].m*k; i++){
                final_result.ndist[results_offset+i] = results_buffer[p_iter-1].ndist[i];
                final_result.nidx[results_offset+i] = results_buffer[p_iter-1].nidx[i];
            }
            //free from the buffer what was just added
            free(results_buffer[p_iter-1].ndist);
            free(results_buffer[p_iter-1].nidx);
            //renew offset with what was just added
            results_offset += results_buffer[p_iter-1].m*k;
        }
    }
    /*
    if(world_rank == 0){
        for(int i=0; i<N; i++){
            for(int j=0; j<k; j++){
                printf("%f  ", final_result.ndist[i*k+j]);
            }
            printf("                  ");
            for(int j=0; j<k; j++){
                printf("%d  ", final_result.nidx[i*k+j]);
            }
            printf("\n");
        }
    }
    */

    MPI_Finalize();
    free(X_blocked);
    free(index_offset);
    free(block_size);
    free(V1_result.ndist);
    free(V1_result.nidx);
    if(world_rank == 0){
        free(corpus_set);
    }
    return 0;
}


double *calculate_distance_matrix(double *X, double *Y, int n, int m, int d){

    double *D = (double *)malloc(n*m*sizeof(double));
    if(D == NULL)
        exit(1);

    double *e = (double *)malloc(d*m*sizeof(double));
    if(e == NULL){
        exit(1);
    }
    for (int i=0; i<(d*m); i++){
        e[i] = 1.0;
    }

    double *X_hadamard = (double *)malloc(n*d*sizeof(double));
    if(X_hadamard == NULL){
        exit(1);
    }
    for (int i=0; i<n*d; i++){
        X_hadamard[i] = X[i]*X[i];
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d,
                1.0, X_hadamard, d, e, d, 0.0, D, m);
    free(e);
    free(X_hadamard);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d,
                -2.0, X, d, Y, d, 1.0, D, m);

    e = (double *)malloc(n*d*sizeof(double));
    if(e == NULL){
        exit(1);
    }
    for (int i=0; i<(n*d); i++){
        e[i] = 1.0;
    }

    double *Y_hadamard = (double *)malloc(m*d*sizeof(double));
    if(Y_hadamard == NULL){
        exit(1);
    }
    for (int query_iter=0; query_iter<(m*d); query_iter++){
        Y_hadamard[query_iter] = Y[query_iter] * Y[query_iter];
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d,
                1.0, e, d, Y_hadamard, d, 1.0, D, m);
    free(e);
    free(Y_hadamard);

    for(int distances_iter=0; distances_iter<(n*m); distances_iter++){
        //float_absolute is used because cblas may produce a -0.00 value
        D[distances_iter] = sqrt(fabs(D[distances_iter]));
    }

    return D;
}

struct knnresult update_knn(struct knnresult result, double *D, int n, int m, int k, int *empty_knn, int index_offset, int block_index_offset){
    double *current_min_distances = (double *)malloc(k*sizeof(double));
    if(current_min_distances == NULL){
        exit(1);
    }
    int *current_min_dist_index = (int *)malloc(k*sizeof(int));
    if(current_min_dist_index == NULL){
        exit(1);
    }
    for(int corpus_iter=0; corpus_iter<n; corpus_iter++){
        for(int nn_iter=0; nn_iter<k; nn_iter++){
            current_min_dist_index[nn_iter] = -1;
            current_min_distances[nn_iter] = INFINITY;
        }
        for(int j=corpus_iter*m; j<(corpus_iter*m + m); j++){
            if(D[j] <= current_min_distances[k-1]){
                current_min_distances[k-1] = D[j];
                current_min_dist_index[k-1] = j - corpus_iter*m + index_offset + block_index_offset;

                // check if sorting is needed after the change
                if(current_min_distances[k-1] < current_min_distances[k-2]){
                    mergeSort_carry_one(current_min_distances, current_min_dist_index, 0, k-1);
                }
            }
        }
        // first neighbors to be added
        if(*empty_knn == 1){
            for(int nn_iter=0; nn_iter<k; nn_iter++){
                result.nidx[nn_iter+k*corpus_iter] = current_min_dist_index[nn_iter];
                result.ndist[nn_iter+k*corpus_iter] = current_min_distances[nn_iter];
            }
        }
        // if not first, then must compare
        else{
            for(int nn_iter=k; nn_iter>0; nn_iter--){
                if (current_min_distances[k - nn_iter] < result.ndist[k*corpus_iter + nn_iter - 1]){
                    result.ndist[k*corpus_iter + nn_iter - 1] = current_min_distances[k - nn_iter];
                    result.nidx[k*corpus_iter + nn_iter - 1] = current_min_dist_index[k - nn_iter];
                }
                else{
                    break;
                }
            }
            mergeSort_carry_one(result.ndist, result.nidx, k*corpus_iter, k*corpus_iter + (k-1));
        }
    }

    if(*empty_knn == 1){
        *empty_knn = 0;
    }
    free(current_min_distances);
    free(current_min_dist_index);
    return result;
}
