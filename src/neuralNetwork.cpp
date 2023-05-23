#include "include/neuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int> &layers)
{
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real_distribution<> distribution(-1, 1);

	weights.resize(layers.size() - 1);
	weightChanges.resize(layers.size() - 1);
	gradients.resize(layers.size() - 1);
	biases.resize(layers.size() - 1);
	biasChanges.resize(layers.size() - 1);
	activations.resize(layers.size());
	deltas.resize(layers.size() - 1);
	errors.resize(layers.size() - 1);
	costDerivatives.resize(layers.size() - 1);

	for (int i = 0; i < (int) weights.size(); ++i)
	{
		weights[i].resize(layers[i + 1]);
		weightChanges[i].resize(layers[i + 1]);
		gradients[i].resize(layers[i + 1]);
		biases[i].resize(layers[i + 1]);
		biasChanges[i].resize(layers[i + 1]);
		deltas[i].resize(layers[i + 1]);
		errors[i].resize(layers[i + 1]);
		costDerivatives[i].resize(layers[i + 1]);

		for (int j = 0; j < layers[i + 1]; ++j)
		{
			weights[i][j].resize(layers[i]);
			weightChanges[i][j].resize(layers[i]);
			gradients[i][j].resize(layers[i]);
			biases[i][j] = distribution(gen);
			biasChanges[i][j] = 0;

			for (int k = 0; k < layers[i]; ++k)
			{
				weights[i][j][k] = distribution(gen);
				weightChanges[i][j][k] = 0;
			}
		}
	}
}

std::vector<double> NeuralNetwork::forward(const std::vector<double> &input)
{
	activations.resize(weights.size() + 1);
	activations[0] = input;

	for (int i = 1; i < (int) activations.size(); ++i)
	{
		activations[i].resize(weights[i - 1].size());
		for (int j = 0; j < (int) activations[i].size(); ++j)
		{
			double sum = 0;
			for (int k = 0; k < (int) activations[i - 1].size(); ++k)
				sum += weights[i - 1][j][k] * activations[i - 1][k];

			activations[i][j] = sigmoid(sum + biases[i - 1][j]);
		}
	}

	return activations.back();
}

void NeuralNetwork::backPropagate(const std::vector<double> &targets)
{
	for (int i = 0; i < (int) activations.size(); ++i)
		errors.back()[i] = costDerivatives.back()[i] * (activations.back()[i] - targets[i]);

	for (int i = (int) errors.size() - 2; i >= 0; --i)
		for (int j = 0; j < (int) errors[i].size(); ++j)
		{
			errors[i][j] = 0;
			for (int k = 0; k < (int) errors[i + 1].size(); ++k)
				errors[i][j] += weights[i + 1][k][j] * errors[i + 1][k];
		}

	for (int i = 0; i < (int) deltas.size(); ++i)
		for (int j = 0; j < (int) deltas[i].size(); ++j)
			deltas[i][j] = errors[i][j] * sigmoidDerivative(activations[i + 1][j]);
}

void NeuralNetwork::updateWeights(double learningRate, double momentum)
{
	for (int i = 0; i < (int) weights.size(); ++i)
		for (int j = 0; j < (int) weights[i].size(); ++j)
		{
			for (int k = 0; k < (int) weights[i][j].size(); ++k)
			{
				gradients[i][j][k] = deltas[i][j] * activations[i][k];
				weightChanges[i][j][k] = learningRate * gradients[i][j][k] + momentum * weightChanges[i][j][k];
				weights[i][j][k] -= weightChanges[i][j][k];
			}

			biasChanges[i][j] = learningRate * deltas[i][j] + momentum * biasChanges[i][j];
			biases[i][j] -= biasChanges[i][j];
		}

}

double NeuralNetwork::getError(const std::vector<double> &target)
{
	double error = 0;
	for (int i = 0; i < (int) target.size(); ++i) error += cost(activations.back()[i], target[i]);

	return error;
}

double NeuralNetwork::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x)
{
	return x * (1 - x);
}

double NeuralNetwork::cost(double output, double target)
{
	return 0.5 * pow(output - target, 2);
}
