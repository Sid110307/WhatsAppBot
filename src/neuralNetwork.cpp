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

std::vector<std::vector<double>> NeuralNetwork::forward(const std::vector<double> &input)
{
	activations.resize(weights.size() + 1);
	activations[0] = input;

	for (int i = 0; i < (int) weights.size(); ++i)
	{
		activations[i + 1].resize(weights[i].size());
		for (int j = 0; j < (int) weights[i].size(); ++j)
		{
			double sum = 0;
			for (int k = 0; k < (int) weights[i][j].size(); ++k)
				sum += weights[i][j][k] * activations[i][k];

			sum += biases[i][j];
			activations[i + 1][j] = sigmoid(sum);
		}
	}

	return activations;
}

void NeuralNetwork::backPropagate(const std::vector<double> &targets)
{
	for (int i = 0; i < (int) deltas.back().size(); ++i)
		deltas.back()[i] = (activations.back()[i] - targets[i]) * sigmoidDerivative(activations.back()[i]);

	for (int i = (int) deltas.size() - 2; i >= 0; --i)
		for (int j = 0; j < (int) deltas[i].size(); ++j)
		{
			double sum = 0;
			for (int k = 0; k < (int) deltas[i + 1].size(); ++k)
				sum += weights[i + 1][k][j] * deltas[i + 1][k];

			deltas[i][j] = sum * sigmoidDerivative(activations[i + 1][j]);
		}
}

void NeuralNetwork::updateWeights(double learningRate, double momentum)
{
	for (int i = 0; i < (int) weights.size(); ++i)
		for (int j = 0; j < (int) weights[i].size(); ++j)
		{
			for (int k = 0; k < (int) weights[i][j].size(); ++k)
			{
				double delta = learningRate * deltas[i][j] * activations[i][k];
				weights[i][j][k] -= weightChanges[i][j][k];
				weights[i][j][k] += delta;
				weightChanges[i][j][k] = delta + momentum * weightChanges[i][j][k];
			}

			double delta = learningRate * deltas[i][j];
			biases[i][j] -= biasChanges[i][j];
			biases[i][j] += delta;
			biasChanges[i][j] = delta + momentum * biasChanges[i][j];
		}
}

double NeuralNetwork::getError(const std::vector<double> &target)
{
	double error = 0;
	for (int i = 0; i < (int) target.size(); ++i) error += cost(activations.back()[i], target[i]);

	return error;
}

bool NeuralNetwork::saveModel(const std::string &filename)
{
	std::ofstream file(filename);
	if (!file || !file.is_open())
	{
		std::cerr << "Error: could not open file " << filename << "\n";
		return false;
	}

	for (int i = 0; i < (int) weights.size(); ++i)
		for (int j = 0; j < (int) weights[i].size(); ++j)
		{
			for (double k: weights[i][j]) file << k << " ";
			file << biases[i][j] << "\n";
		}

	file.close();
	return true;
}

bool NeuralNetwork::loadModel(const std::string &filename)
{
	std::ifstream file(filename);
	if (!file || !file.is_open())
	{
		std::cerr << "Error: could not open file " << filename << "\n";
		return false;
	}

	for (int i = 0; i < (int) weights.size(); ++i)
		for (int j = 0; j < (int) weights[i].size(); ++j)
		{
			for (double &k: weights[i][j]) file >> k;
			file >> biases[i][j];
		}

	file.close();
	return true;
}

double NeuralNetwork::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x)
{
	auto sigmoidX = sigmoid(x);
	return sigmoidX * (1 - sigmoidX);
}

double NeuralNetwork::cost(double output, double target)
{
	return 0.5 * pow(output - target, 2);
}
