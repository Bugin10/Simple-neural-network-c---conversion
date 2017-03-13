#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include <stdlib.h>    
#include <time.h>

using Eigen::MatrixXd;
using namespace std;

MatrixXd sigmoid(MatrixXd x)
{
    MatrixXd expReturn(x.rows(), x.cols());
    for(int i=0; i < x.rows(); i++)
    {
        expReturn(i) = 1/(1 + exp(-x(i)));
    }
    return expReturn;
}

MatrixXd sigmoidDerivative(MatrixXd x)
{
    MatrixXd oneMatrix = MatrixXd::Constant(x.rows(), x.cols(), 1);
    return  x.cwiseProduct(oneMatrix - x);
}

class NeuralNetwork
{
   
    public:
    MatrixXd synapticWeights;
    NeuralNetwork() : synapticWeights(3,1)
    {
        srand (time(NULL));
        synapticWeights(0) = (rand() % 100000 - 50000) / 100000.0;
        synapticWeights(1) = (rand() % 100000 - 50000) / 100000.0;
        synapticWeights(2) = (rand() % 100000 - 50000) / 100000.0;
    }

    void train( MatrixXd trainingSetInputs, MatrixXd trainingSetOutputs, int numberOfTrainingIterations)
    {
        for (int i=0; i<numberOfTrainingIterations; i++)
        {
            MatrixXd output = think(trainingSetInputs);
            MatrixXd error = trainingSetOutputs - output;
            MatrixXd adjustment = trainingSetInputs.transpose() * (error.cwiseProduct(sigmoidDerivative(output)));
            synapticWeights += adjustment;
        }
    }

    MatrixXd think(MatrixXd inputs)
    {
        return sigmoid(inputs*synapticWeights);
    }
};




int main()
{
    MatrixXd trainingSetInputs(4,3);
    trainingSetInputs << 0,0,1,1,1,1,1,0,1,0,1,1;
    MatrixXd trainingSetOutputs(4,1);
    trainingSetOutputs << 0,1,1,0;

    NeuralNetwork neural_network = NeuralNetwork();

    cout<<"Random starting synaptic weights: "<<endl;
    cout<<neural_network.synapticWeights<<endl<<endl;
    

    neural_network.train(trainingSetInputs, trainingSetOutputs, 1000);
    cout<<"New synaptic weights after training: "<<endl;
    cout<<neural_network.synapticWeights<<endl<<endl;

    cout<<"Considering new situation [1, 0, 0] -> ?: "<<endl;
    MatrixXd testcase(1,3);
    testcase<<1,0,0;
    cout<<neural_network.think(testcase);
}