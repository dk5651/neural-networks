#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

// Activation function: sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
private:
    vector<vector<double>> weights;
    vector<double> biases;
    double learning_rate;

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate) {
        this->learning_rate = learning_rate;

        // Initialize weights randomly
        srand(time(0));
        for (int i = 0; i < hidden_size; ++i) {
            vector<double> layer_weights;
            for (int j = 0; j < input_size; ++j) {
                layer_weights.push_back((double)rand() / RAND_MAX);
            }
            weights.push_back(layer_weights);
        }

        // Initialize biases randomly
        for (int i = 0; i < hidden_size; ++i) {
            biases.push_back((double)rand() / RAND_MAX);
        }
    }

    vector<double> forward(vector<double>& input) {
        vector<double> hidden(weights.size());

        // Compute hidden layer activations
        for (int i = 0; i < weights.size(); ++i) {
            double activation = biases[i];
            for (int j = 0; j < weights[i].size(); ++j) {
                activation += input[j] * weights[i][j];
            }
            hidden[i] = sigmoid(activation);
        }

        return hidden;
    }

    void train(vector<double>& input, vector<double>& target) {
        vector<double> hidden = forward(input);

        // Backpropagation
        for (int i = 0; i < weights.size(); ++i) {
            double error = target[i] - hidden[i];
            double delta = error * sigmoid_derivative(hidden[i]);

            // Update weights
            for (int j = 0; j < weights[i].size(); ++j) {
                weights[i][j] += learning_rate * delta * input[j];
            }

            // Update biases
            biases[i] += learning_rate * delta;
        }
    }
};

int main() {
    // Example usage
    NeuralNetwork nn(2, 3, 1, 0.1); // 2 input nodes, 3 hidden nodes, 1 output node, learning rate 0.1

    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> y = {{0}, {1}, {1}, {0}};

    // Training loop
    for (int epoch = 0; epoch < 10000; ++epoch) {
        for (int i = 0; i < X.size(); ++i) {
            nn.train(X[i], y[i]);
        }
    }

    // Testing
    for (int i = 0; i < X.size(); ++i) {
        vector<double> prediction = nn.forward(X[i]);
        cout << "Input: " << X[i][0] << ", " << X[i][1] << " - Prediction: " << prediction[0] << endl;
    }

    return 0;
}
