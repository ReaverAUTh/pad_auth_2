#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "VPT.h"
#include "VPT.c"
#include "supplementary.h"
#include "supplementary.c"
#include <mpi.h>

void searchVPT(vptree *T, double *point, double *point_dist, int *point_index, int d, int k);
void update_knn(double * vp, int vp_index, double *point, double *point_dist, int *point_index, double *distance, int d, int k);


//! Global variables

// block size for each process (size: m*d)
int *block_size;

//! Global varialbes end

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
struct knnresult distrAllkNN(double * X, int n, int d, int k){
    // result will contain the neighbors of the corpus_block of each process
    // that was shared to it by world_rank=0
    // therefore the nidx, ndist arrays must be of size n*k
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

    // value initialization
    for(int i=0; i<(n*k); i++){
        result.nidx[i] = -1;
        result.ndist[i] = INFINITY;
    }

    // MPI data
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int processes_count;
    MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

	int m = result.m;
	int rcved_query_count;
	int empty_knn = 1;

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

        struct VPtree *T = createVPT(Y, m, d, index_offset[world_rank]);

        //! Iterate through all corpus_block points
        // point_index contains the indexes of its kn-neighbors
        // point_dist contains the distances of its kn-neighbors
        // point contains the point's coordinates

        // the iteration
        
        for(int points_iter=0; points_iter<n; points_iter++){
            double *point = (double *)malloc(d*sizeof(double));
            if (point == NULL){
                exit(1);
            }
            int *point_index = (int *)malloc(k*sizeof(int));
            if (point_index == NULL){
                exit(1);
            }
            double *point_dist = (double *)malloc(k*sizeof(double));
            if (point_dist == NULL){
                exit(1);
            }

            for(int coord_iter=0; coord_iter<d; coord_iter++){
                point[coord_iter] = X[points_iter*d+coord_iter];
            }
            for(int nn_iter=0; nn_iter<k; nn_iter++){
                point_dist[nn_iter] = result.ndist[points_iter*k + nn_iter];
                point_index[nn_iter] = result.nidx[points_iter*k + nn_iter];
            }

            searchVPT(T, point, point_dist, point_index, d, k);
            free(point);

            for(int j=0; j<k; j++){
                result.ndist[points_iter*k + j] = point_dist[j];
                result.nidx[points_iter*k + j] = point_index[j];
            }
            free(point_index); free(point_dist);
        }
        destroy(T);

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

    struct knnresult V2_result = distrAllkNN(X_blocked,n,d,k);

    // wait for everything to get here, so that process 0 can count
    // the duration correctly and accumulate all data
    MPI_Barrier(MPI_COMM_WORLD);

    struct timespec toc;
    if(world_rank ==0){
          clock_gettime( CLOCK_MONOTONIC, &toc);
          printf("V2 Duration = %f sec\n", time_spent(tic, toc));
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
            final_result.nidx[i] = V2_result.nidx[i];
            final_result.ndist[i] = V2_result.ndist[i];
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
            MPI_Send(V2_result.ndist, n*k, MPI_DOUBLE, 0, p_iter, MPI_COMM_WORLD);
        }
        else if(world_rank == 0){
            MPI_Recv(results_buffer[p_iter-1].ndist, results_buffer[p_iter-1].m*k, MPI_DOUBLE, p_iter, p_iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // nidx S/R

        if(world_rank == p_iter){
            MPI_Send(V2_result.nidx, n*k, MPI_INT, 0, p_iter, MPI_COMM_WORLD);
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
    free(V2_result.ndist);
    free(V2_result.nidx);
    if(world_rank == 0){
        free(corpus_set);
    }
    return 0;
}


void searchVPT(struct VPtree *T, double *point, double *point_dist, int *point_index, int d, int k){

    if(T == NULL){
        return;
    }
    // distance = dist between point and vantage point
    double distance = 0;
    // check for new neighbor
    update_knn(getVP(T), getIDX(T), point, point_dist, point_index, &distance, d, k);
    double max_dist = point_dist[k-1];

    if (getInner(T) == NULL && getOuter(T) == NULL){
        return;
    }

    bool intersection = (distance + max_dist >= getMD(T));

    if(distance < getMD(T)){
        if(distance - max_dist <= getMD(T)){
            searchVPT(getInner(T), point, point_dist, point_index, d, k);
        }
        if(intersection){
            searchVPT(getOuter(T), point, point_dist, point_index, d, k);
        }
    }
    else{
        if(intersection){
            searchVPT(getOuter(T), point, point_dist, point_index, d, k);
        }
        if(distance - max_dist <= getMD(T)){
            searchVPT(getInner(T), point, point_dist, point_index, d, k);
        }
    }
}


void update_knn(double *vp, int vp_index, double *point, double *point_dist, int *point_index, double *distance, int d, int k){
    // calculate euclidean distance
    for (int i=0; i<d; i++){
        double coord_diff = vp[i]-point[i];
        *distance += coord_diff*coord_diff;
    }
    *distance = sqrt(fabs(*distance));

    if (*distance < point_dist[k-1]){
        point_dist[k-1] = *distance;
        point_index[k-1] = vp_index;

        // check if sorting is needed after the change
        if(point_dist[k-1] < point_dist[k-2]){
            mergeSort_carry_one(point_dist, point_index, 0, k-1);
        }
    }
}
