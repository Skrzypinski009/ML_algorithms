#include "neural_network.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>

double RandomGenerator()
{
    double min = -1, max = 1;
    double rand = (double)std::rand()/RAND_MAX;
    return rand * (max - min) + min;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes)
{
    Create(layer_sizes);
}

NeuralNetwork::NeuralNetwork(std::string path)
{
    Load(path);
}

NeuralNetwork::~NeuralNetwork()
{
    Clear();
}

void NeuralNetwork::Create(const std::vector<int>& layer_sizes = {})
{
    Neurons = {};
    Weights = {};
    Ndelta = {};
    Wdelta = {};

    if (layer_sizes.size() > 0)
        LayerSizes = layer_sizes;
    // fill Neurons vector
    for (int i=0; i<LayerSizes.size(); i++) 
    {
        std::vector<double> vec(LayerSizes[i], false);
        Neurons.push_back(vec);
    }
    // fill Weights vector
    for (int i=1; i<LayerSizes.size(); i++)
    {
        std::vector<std::vector<double>> vec;

        for (int j=0; j<LayerSizes[i]; j++)
        {
            std::vector<double> prev (LayerSizes[i-1] + 1, 0); // + 1 for bias
            // std::generate(prev.begin(), prev.end(), RandomGenerator);
            vec.push_back(prev);
        }
        Weights.push_back(vec);
    }
    // setting up delta vector for Neurons
    for (int i=0; i<LayerSizes.size(); i++)
    {
        std::vector<double> vec(LayerSizes[i], 0);
        Ndelta.push_back(vec);
    }
    // setting up delta vector for Weights
    for (int i=0; i<Weights.size(); i++)
    {
        std::vector<std::vector<double>> layer;
        for (int j=0; j<Weights[i].size(); j++)
        {
            std::vector<double> vec(Weights[i][j].size(), 0);
            layer.push_back(vec);
        }
        Wdelta.push_back(layer);
    }
}

void NeuralNetwork::SetInput(std::vector<double> input)
{
    if (Neurons[0].size() != input.size()) return;
    Neurons[0] = input;
}


double NeuralNetwork::SFunction(int i, int j)
{
    double s = Weights[i-1][j][0]; // weight[0] is bias
    for (int k=1; k<Neurons[i-1].size() + 1; k++) // neurons <1; last> in layer 0
    {
        s += Neurons[i-1][k] * Weights[i-1][j][k];
    }
    return s;
}

double NeuralNetwork::dSFunctionWeight(int i, int j)
{
    return Neurons[i-1][j];
}

double NeuralNetwork::dSFunctionBias()
{
    return 1;
}

double NeuralNetwork::SigmoidFunction(double s)
{
    return 1 / (1 + exp(-s));
}

double NeuralNetwork::dSigmoidFunction(double s)
{
    return exp(s) / (exp(2*s) + 2 * exp(s) + 1);
}

double NeuralNetwork::Error(double expected_output, double output)
{
    return pow(expected_output - output, 2);
}

double NeuralNetwork::dError(double expected_output, double output)
{
    return 2 * (expected_output - output);
}

double NeuralNetwork::NetworkError(std::vector<double> expected_output)
{
    double error = 0;
    for (int i=0; i<Neurons[Neurons.size()-1].size(); i++)
    {
        error += Error(expected_output[i],Neurons[Neurons.size()-1][i]);
    }
    return error;
}

void NeuralNetwork::ForwardPropagation()
{
    for (int i=1; i<Neurons.size(); i++) // layers <1: last>
    {
        for (int j=0; j<Neurons[i].size(); j++) // neurons <0; last> in layer 1
        {
            double s = SFunction(i, j);
            Neurons[i][j] = SigmoidFunction(s);
        }
    }
}

void NeuralNetwork::BackwardPropagation(std::vector<double> expected_output)
{
    for (int i=LayerSizes.size()-1; i>0; i--) // for all layers except first
    {
        for (int j=0; j<LayerSizes[i]; j++) // for size in each layer
        {
            // neurons delta
            if (i==LayerSizes.size()-1) // is it the last layer
            {
                Ndelta[i][j] = dError(expected_output[j], Neurons[i][j]);
            }
            else
            {
                for (int k=0; k<Neurons[i+1].size(); k++)
                    Ndelta[i][j] += Weights[i][k][j+1] * dSigmoidFunction(SFunction(i+1, k)) * Ndelta[i+1][k];
            }
            // bias delta
            Wdelta[i-1][j][0] = dSigmoidFunction(SFunction(i, j)) * Ndelta[i][j];
            // updating bias
            Weights[i-1][j][0] += Wdelta[i-1][j][0];
            //weights delta
            for (int k=0; k<Neurons[i-1].size(); k++)
            {
                Wdelta[i-1][j][k+1] = Neurons[i-1][k] * dSigmoidFunction(SFunction(i, j)) * Ndelta[i][j];
                // updating weights
                Weights[i-1][j][k+1] += Wdelta[i-1][j][k+1];
            }
        }
    }
}

void NeuralNetwork::FillWeightsRandom()
{
    for (auto& layer : Weights)
    {
        for (auto& neuron : layer)
        {
            std::generate(neuron.begin(), neuron.end(), RandomGenerator);
        }
    }
}

void NeuralNetwork::Save(std::string path) const
{
    std::ofstream file(path);

    file << LayerSizes.size() << "\n";
    for (int size : LayerSizes)
        file << size << " ";
    
    file << "\n";
    
    for (auto layer : Weights)
    {
        for (auto neuron : layer)
        {
            for (double weight : neuron)
                file << weight << " ";

            file << "\n";
        }
        file << "\n";
    }

    file.close();
}

void NeuralNetwork::Load(std::string path)
{
    Clear();

    int layers = 0;
    int size_of_layer = 0;
    std::ifstream file(path);
    file >> layers;
    
    for (int i=0; i<layers; i++)
    {
        file >> size_of_layer;
        LayerSizes.push_back(size_of_layer);
    }
    Create();

    for (auto& layer : Weights)
    {
        for (auto& neuron : layer)
        {
            for (double& weight : neuron)
                file >> weight;
        }
    }

    file.close();
}

void NeuralNetwork::Clear() 
{
    for (auto& layer : Neurons)
    {
        layer.clear();
    }
    Neurons.clear();

    for (auto& layer : Ndelta)
    {
        layer.clear();
    }
    Ndelta.clear();

    for (auto& layer : Weights)
    {
        for (auto& neuron : layer)
        {
            neuron.clear();
        }
        layer.clear();
    }
    Weights.clear();

    for (auto& layer : Wdelta)
    {
        for (auto& neuron : layer)
        {
            neuron.clear();
        }
        layer.clear();
    }
    Wdelta.clear();

    LayerSizes.clear();
}

void NeuralNetwork::Print()
{
    std::cout << "\n---NEURONS---" << std::endl;
    for(auto layer : Neurons)
    {
        std::cout<<"[";
        for(double val : layer)
            std::cout<< val << ", ";
        std::cout<<"]\n";
    }

    std::cout << "\n\n---WEIGHTS---" << std::endl;
    for (auto layer : Weights)
    {
        std::cout << "[\n";
        for (auto neuron : layer)
        {
            std::cout << "    [";
            for (double w : neuron)
                std::cout << w << ", ";
            std::cout << "]\n";
        }
        std::cout << "]" << std::endl;
    }

}

void NeuralNetwork::PrintOutput()
{
    std::cout << "OUTPUT: [";
    for (double value : Neurons[Neurons.size()-1])
    {
        std::cout << value << ", ";
    }
    std::cout << "]" << std::endl;
}
