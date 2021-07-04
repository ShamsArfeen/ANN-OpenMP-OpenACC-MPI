# ANN-OpenMP-OpenACC-MPI
A generic implementation of feedforward algorithm in OpenMP, OpenACC, MPI and a combined approach in C, with time linear to the number of edges. Every layer has been assumed to have equal number of neurons - the hidden layers as well as the input and output layers. The size of layers and the number of intermediate layers can be controlled with compile-time constants namely, BREADTH and DEPTH. The constants represent the count of nodes per layer and the count of all layers, respectively. With increasing parallelism, BREADTH will have decreased effect on execution time. Activation function (tanh()) is also the same for all layers except that it can be changed to any other sigmoid function.

