#pragma once

#include <iostream>
#include <fstream>
#include <random>

class NeuralNetwork
{
public:
	explicit NeuralNetwork(const std::vector<int> &);
	std::vector<std::vector<double>> forward(const std::vector<double> &);
	void backPropagate(const std::vector<double> &);

	void updateWeights(double, double);
	double getError(const std::vector<double> &);

	bool saveModel(const std::string &);
	bool loadModel(const std::string &);

private:
	std::vector<std::vector<std::vector<double>>> weights, weightChanges, gradients;
	std::vector<std::vector<double>> biases, biasChanges;
	std::vector<std::vector<double>> activations, deltas, errors, costDerivatives;

	static double sigmoid(double);
	static double sigmoidDerivative(double);
	static double cost(double, double);
};
