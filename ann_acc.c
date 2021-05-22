#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define DEPTH 2 // no of layers
#define BREADTH 1000 // neurons per layer
#define OUTPUT 1000 // output neurons
#define INPUT 1000 // input neurons

struct unit {
    double active;
    double charge;
};

double start[INPUT][BREADTH];
double weight[DEPTH][BREADTH][BREADTH];
double final[INPUT][BREADTH];
struct unit neuron[DEPTH][BREADTH];
struct unit sequence[OUTPUT];


double input[BREADTH];
struct unit output[OUTPUT];
int feedforward() {
    input[0] = 1;
    
    double sum; // created to be used later in loops within the reduction clause to parallelize code

    // parallelize all the arrays to reduce time taken
    #pragma acc enter data present_or_copyin(output, neuron, input, start, final)
    
    // open acc parallel directive is being implemented to gain maximum speedup
    #pragma acc parallel
    {   
        // followed by the initial parallel directive
        #pragma acc loop
        for ( int i = 1; i < BREADTH; i++ )
        {   
            // sum is re-initialized within loop so value can be reset for calculation with each iteration
            sum = 0;
            
            // reduction clause is used to stop loop iterations to write directly to sum simultaneously
            #pragma acc loop reduction(+:sum)
            for ( int j = 0; j < INPUT; j++ )
            {
                sum += start[j][i] * input[j];
            }
            neuron[0][i].charge = sum;
            neuron[0][i].active = tanh( neuron[0][i].charge );
        }
        // neuron[0][0].active = 1;
    
        for ( int k = 1; k < DEPTH ; k++ )
        {
            // followed by the initial parallel directive
            #pragma acc loop
            for ( int i = 1; i < BREADTH ; i++ )
            {
                // sum is re-initialized within loop so value can be reset for calculation with each iteration
                sum = 0;

                // reduction clause is used to stop loop iterations to write directly to sum simultaneously
                #pragma acc loop reduction(+:sum)
                for ( int j = 0; j < BREADTH; j++ )
                {
                    sum += weight[k-1][j][i] * neuron[k-1][j].active;
                }
                neuron[k][i].charge = sum;
                neuron[k][i].active = tanh( neuron[k][i].charge );
            }
            // neuron[k][0].active = 1;
        }
    
        // followed by the initial parallel directive
        #pragma acc loop
        for ( int i = 0; i < OUTPUT; i++ )
        {   
            // sum is re-initialized within loop so value can be reset for calculation with each iteration
            sum = 0;

            // reduction clause is used to stop loop iterations to write directly to sum simultaneously
            #pragma acc loop reduction(+:sum)
            for ( int j = 0; j < BREADTH; j++ )
            {
                sum += final[j][i] * neuron[DEPTH-1][j].active;
            }
            output[i].charge = sum;
            output[i].active = tanh( output[i].charge );
        }
    }
    
    return 0;
    
}

int main() {
    
	int ssec, esec, susec, eusec;
	struct timeval tv;
  
	gettimeofday(&tv, NULL);
	ssec = tv.tv_sec;
	susec = tv.tv_usec;

    for ( int i = 0; i < 100; i++)
        feedforward();

	gettimeofday(&tv, NULL);
  	esec = tv.tv_sec;
	eusec = tv.tv_usec;
    
	double dtime = ((esec * 1.0) + ((eusec * 1.0)/1000000.0)) - ((ssec * 1.0) + ((susec * 1.0)/1000000.0));
	printf("TIME %.3f\n", dtime);

    return 0;
}