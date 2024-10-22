#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "neural_network.h"

int main()
{
    std::srand(time(0));
    std::vector<int> layer_sizes = {4,6,3};
    NeuralNetwork nn(layer_sizes);
    nn.FillWeightsRandom();
    nn.Save("nn.txt");
    
    // training;
    std::vector<double> input = {0,0,1,1};
    std::vector<double> expected_output = {0.056 ,0.42, 0.1};
    nn.SetInput(input);
    for (int i=0; i<200; i++)
    {
        nn.ForwardPropagation();
        std::cout<< nn.NetworkError(expected_output) << std::endl;
        nn.PrintOutput();
        nn.BackwardPropagation(expected_output);
    }
    return 0;
}

