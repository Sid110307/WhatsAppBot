#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>

class NeuralNetwork
{
public:
	explicit NeuralNetwork(const std::vector<int> &layers);
	std::vector<std::vector<double>> forward(const std::vector<double> &input);
	void backPropagate(const std::vector<double> &targets);

	void updateWeights(double learningRate, double momentum);
	double getError(const std::vector<double> &target);

	bool saveModel(const std::string &filename);
	bool loadModel(const std::string &filename);

private:
	std::vector<std::vector<std::vector<double>>> weights, weightChanges, gradients;
	std::vector<std::vector<double>> biases, biasChanges;
	std::vector<std::vector<double>> activations, deltas, errors, costDerivatives;

	static double sigmoid(double x);
	static double sigmoidDerivative(double x);
	static double cost(double output, double target);
};
