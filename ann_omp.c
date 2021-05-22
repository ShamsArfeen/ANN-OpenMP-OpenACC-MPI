#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <stdlib.h>

#define DEPTH 2 // no of layers
#define BREADTH 800 // neurons per layer
#define OUTPUT 800 // output neurons
#define INPUT 800 // input neurons

#define THREADNUM 10

struct unit { // neuron unit
    double active;
    double charge;
};

double start[INPUT][BREADTH]; // weight matrix of first layer
double weight[DEPTH][BREADTH][BREADTH];  // weight matrix of intermediate layers
double final[BREADTH][OUTPUT]; // weight matrix of last layer
struct unit neuron[DEPTH][BREADTH]; // neural network

double input[BREADTH]; // input vector
struct unit output[OUTPUT]; // output neurons
int feedforward() {
    input[0] = 1;

    #pragma omp parallel for
    for ( int i = 1; i < BREADTH ; i++ ) { // independent loop
        for ( int j = 0; j < INPUT; j++ ) { // output carried dependency
            neuron[0][i].charge += start[i][j] * input[j]; // SAXPY form row major
        }
        neuron[0][i].active = tanh( neuron[0][i].charge ); // activation function
    }
    neuron[0][0].active = 1;

    for ( int k = 1; k < DEPTH ; k++ ) {
        #pragma omp parallel for
        for ( int i = 1 ; i < BREADTH ; i++ ) { // independent loop
            for ( int j = 0; j < BREADTH; j++ ) { // output carried dependency
                neuron[k][i].charge += weight[k-1][i][j] * neuron[k-1][j].active; // SAXPY form row major
            }
            neuron[k][i].active = tanh( neuron[k][i].charge ); // activation function
        }
        neuron[k][0].active = 1;
    }
    
    #pragma omp parallel for
    for ( int i = 1 ; i < OUTPUT ; i++ ) { // independent loop
        for ( int j = 0; j < BREADTH; j++ ) { // output carried dependency
            output[i].charge += final[i][j] * neuron[DEPTH-1][j].active; // SAXPY form row major
        }
        output[i].active = tanh( output[i].charge ); // activation function
    }
    return 0;
}

int init() { // array random init
    
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

int main() {

    init();
    
	int ssec, esec, susec, eusec;
	struct timeval tv;
    
	omp_set_num_threads(THREADNUM); // setting thread num

	gettimeofday(&tv, NULL); // task start
	ssec = tv.tv_sec;
	susec = tv.tv_usec;

    for (int i = 0; i < 100; i++) // 100 data points
        feedforward();

	gettimeofday(&tv, NULL); // task finish
	esec = tv.tv_sec;
	eusec = tv.tv_usec;
    

    // Execution time
	double dtime = ((esec * 1.0) + ((eusec * 1.0)/1000000.0)) - ((ssec * 1.0) + ((susec * 1.0)/1000000.0));
	printf("TIME %.3f\n", dtime);

    return 0;
}