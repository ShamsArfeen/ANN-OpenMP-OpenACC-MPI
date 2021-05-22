#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include <omp.h>
#include <stdlib.h>
// mpicc ann_mpi.c -lm -fopenmp
// mpirun -n 4 ./a.out

#define DEPTH 2 // no of layers
#define BREADTH 800 // neurons per layer
#define OUTPUT 800 // output neurons
#define INPUT 800 // input neurons

#define THREADNUM 10

struct unit {
    double active;
    double charge;
};

int rank, size;

double start[INPUT][BREADTH]; // weight matrix of first layer
double weight[DEPTH][BREADTH][BREADTH];  // weight matrix of intermediate layers
double final[BREADTH][OUTPUT]; // weight matrix of last layer
struct unit neuron[DEPTH][BREADTH]; // neural network

double input[BREADTH];
struct unit output[OUTPUT];
int feedforward() {
    input[0] = 1;

	int first = (rank * INPUT) / size;
	int end = ((rank+1) * INPUT) / size;
    #pragma omp parallel for
    for ( int i = first; i < end; i++ ) { // parallelizing the independent loop
        for ( int j = 0; j < INPUT; j++ ) { // dot product saxpy form in chunk
            neuron[0][i].charge += start[i][j] * input[j];
        }
        neuron[0][i].active = tanh( neuron[0][i].charge );
    }
    for (int i = 0; i < size; i++) { // every process will broadcast one after another
		first = (i * INPUT) / size; // start of chunk
		end = ((i+1) * INPUT) / size; // end of chunk
    	MPI_Bcast( &neuron[0][first], 2*(end-first), MPI_DOUBLE, i, MPI_COMM_WORLD); // synchronize chunk
    }
    neuron[0][0].active = 1;

    for ( int k = 1; k < DEPTH ; k++ ) {	
		first = (rank * BREADTH) / size;
		end = ((rank+1) * BREADTH) / size;
        #pragma omp parallel for
        for ( int i = first ; i < end ; i++ ) { // parallelizing the independent loop
            for ( int j = 0; j < BREADTH; j++ ) { // dot product saxpy form in chunk
                neuron[k][i].charge += weight[k-1][i][j] * neuron[k-1][j].active;
            }
            neuron[k][i].active = tanh( neuron[k][i].charge );
        }
		for (int i = 0; i < size; i++) { // every process will broadcast one after another
			first = (i * BREADTH) / size; // start of chunk
			end = ((i+1) * BREADTH) / size; // end of chunk
			MPI_Bcast( &neuron[k][first], 2*(end-first), MPI_DOUBLE, i, MPI_COMM_WORLD); // synchronize chunk
		}
        neuron[k][0].active = 1;
    }
    
	first = (rank * OUTPUT) / size;
	end = ((rank+1) * OUTPUT) / size;
    #pragma omp parallel for
    for ( int i = first ; i < end ; i++ ) {  // parallelizing the independent loop
        for ( int j = 0; j < BREADTH; j++ ) { // dot product saxpy form in chunk
            output[i].charge += final[i][j] * neuron[DEPTH-1][j].active;
        }
        output[i].active = tanh( output[i].charge );
    }
	for (int i = 0; i < size; i++) { // every process will broadcast one after another
		first = (i * OUTPUT) / size; // start of chunk
		end = ((i+1) * OUTPUT) / size; // end of chunk
		MPI_Bcast( &output[first], 2*(end-first), MPI_DOUBLE, i, MPI_COMM_WORLD); // synchronize chunk
	}
    return 0;
}

int init() {
    
	srand( 3);
	
	for( int i=0; i<DEPTH; i++)
		for( int j=0; j<BREADTH; j++)
		    for( int k=0; k<BREADTH; k++) {
			    weight[i][k][j] = (double)((10 - (rand() % 10)) * 0.01);
		    }
            
	for( int i=0; i<INPUT; i++)
		for( int j=0; j<BREADTH; j++) {
			start[j][i] = (double)((10 - (rand() % 10)) * 0.01);
		}

	for( int i=0; i<BREADTH; i++)
		for( int j=0; j<OUTPUT; j++) {
			final[j][i] = (double)((10 - (rand() % 10)) * 0.01);
		}
        
	for( int i=0; i<BREADTH; i++)
        input[i] = (double)((10 - (rand() % 10)) * 0.01);

    return 0;
}

int main (int argc, char** argv) {

	MPI_Init( &argc, &argv);
	MPI_Comm_rank( MPI_COMM_WORLD, &rank);
	MPI_Comm_size( MPI_COMM_WORLD, &size);

    init();
    
	omp_set_num_threads(THREADNUM); // setting thread num
    
	int ssec, esec, susec, eusec;
	struct timeval tv;

	gettimeofday(&tv, NULL); // task start
	ssec = tv.tv_sec;
	susec = tv.tv_usec;

    for (int i = 0; i < 100; i++) // 100 data points
        feedforward();

	gettimeofday(&tv, NULL); // task finish
	esec = tv.tv_sec;
	eusec = tv.tv_usec;

    if (rank == 0) 
	{	
		// Execution time
		double dtime = ((esec * 1.0) + ((eusec * 1.0)/1000000.0)) - ((ssec * 1.0) + ((susec * 1.0)/1000000.0));
    	printf("TIME %.3f\n", dtime);
	}
	MPI_Finalize();

    return 0;
}
