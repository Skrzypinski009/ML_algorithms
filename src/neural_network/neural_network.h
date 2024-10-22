#include <vector>
#include <string>

class NeuralNetwork {
    std::vector<int> LayerSizes;
    std::vector<std::vector<double>> Neurons;
    std::vector<std::vector<std::vector<double>>> Weights;
    std::vector<std::vector<double>> Ndelta;
    std::vector<std::vector<std::vector<double>>> Wdelta;

    double SFunction(int i, int j);
    double dSFunctionWeight(int i, int j);
    double dSFunctionBias();
    double SigmoidFunction(double s);
    double dSigmoidFunction(double s);
    double Error(double expected_output, double output);
    double dError(double expected_output, double output);
public:
    NeuralNetwork(const std::vector<int>& layer_sizes);
    NeuralNetwork(std::string path);
    ~NeuralNetwork();

    void Create(const std::vector<int>& layer_sizes);
    void SetInput(std::vector<double> input);
    void ForwardPropagation();
    void BackwardPropagation(std::vector<double> expected_output);
    void FillWeightsRandom();

    double NetworkError(std::vector<double> expected_output);

    void Save(std::string path) const;
    void Load(std::string path);
    void Clear();
    
    void Print();
    void PrintOutput();
};
