#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "include/neuralNetwork.h"

#define LEARNING_RATE 0.1
#define MOMENTUM 0.9
#define EPOCHS 100
#define ERROR_THRESHOLD 0.01
#define MAX_RESPONSES 5

#ifndef USER
#error "Please define USER when compiling (e.g. -DUSER=\"John Doe\"). The user is the person whose messages will be used to train the neural network."
#endif

struct Message
{
	std::string sender;
	std::string content;
};

std::vector<Message> readChat(const std::string &filename)
{
	std::vector<Message> messages;
	std::ifstream file(filename);
	std::string line;

	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << filename << std::endl;
		return messages;
	}

	while (std::getline(file, line))
	{
		if (line.empty()) continue;
		if (line.find("\u200E") != std::string::npos) continue;

		size_t closingBracketPos = line.find(']');
		size_t colonPos = line.find(':');

		if (closingBracketPos == std::string::npos || colonPos == std::string::npos)
			continue;

		Message message;
		message.sender = line.substr(closingBracketPos + 2, colonPos - closingBracketPos - 2);
		message.content = line.substr(colonPos + 2);
		messages.push_back(message);
	}

	file.close();
	return messages;
}

std::vector<std::string> tokenize(const std::string &str)
{
	std::vector<std::string> tokens;
	std::string token;

	for (char c: str)
	{
		if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?')
		{
			if (!token.empty()) tokens.push_back(token);
			token.clear();
		} else token += c;
	}

	if (!token.empty()) tokens.push_back(token);
	return tokens;
}

std::vector<std::string> getVocabulary(const std::vector<Message> &messages)
{
	std::vector<std::string> vocabulary;
	for (const Message &message: messages)
	{
		std::vector<std::string> tokens = tokenize(message.content);
		for (const std::string &token: tokens)
		{
			bool found = false;
			for (const std::string &word: vocabulary)
				if (token == word)
				{
					found = true;
					break;
				}

			if (!found) vocabulary.push_back(token);
		}
	}

	return vocabulary;
}

std::vector<std::vector<double>> getInputs(const std::vector<Message> &messages, const std::vector<std::string> &vocab)
{
	std::vector<std::vector<double>> inputs;
	for (const Message &message: messages)
	{
		std::vector<double> input(vocab.size());
		std::vector<std::string> tokens = tokenize(message.content);
		for (const std::string &token: tokens)
		{
			for (int i = 0; i < (int) vocab.size(); ++i)
				if (token == vocab[i])
				{
					input[i] = 1;
					break;
				}
		}

		inputs.push_back(input);
	}

	return inputs;
}

std::vector<std::vector<double>> getOutputs(const std::vector<Message> &messages, const std::vector<std::string> &vocab)
{
	std::vector<std::vector<double>> outputs;
	for (const Message &message: messages)
	{
		std::vector<double> output(vocab.size());
		for (int i = 0; i < (int) vocab.size(); ++i) if (message.sender == USER) output[i] = 1;

		outputs.push_back(output);
	}

	return outputs;
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <chat file>" << std::endl;
		std::cout << "To export a chat from WhatsApp, open the chat, tap the three dots in the top right corner, tap "
				  << R"("More", tap "Export chat", and select "Without media".)" << std::endl;
		return EXIT_FAILURE;
	}

	std::vector<Message> messages = readChat(argv[1]);
	if (messages.empty()) return EXIT_FAILURE;

	std::vector<std::string> vocabulary = getVocabulary(messages);
	std::vector<std::vector<double>> inputs = getInputs(messages, vocabulary);
	std::vector<std::vector<double>> outputs = getOutputs(messages, vocabulary);

	std::vector<int> layers = {(int) vocabulary.size(), 10, 10, (int) vocabulary.size()};
	NeuralNetwork nn(layers);

	for (int i = 0; i < EPOCHS; ++i)
	{
		double error = 0;
		for (int j = 0; j < (int) inputs.size(); ++j)
		{
			nn.forward(inputs[j]);
			nn.backPropagate(outputs[j]);
			nn.updateWeights(LEARNING_RATE, MOMENTUM);
			error += nn.getError(outputs[j]);
		}

		error /= (double) inputs.size();
		if (error < ERROR_THRESHOLD)
		{
			std::cout << "Error threshold reached after " << i + 1 << " epochs." << std::endl;
			break;
		}
	}

	std::string message;
	std::cout << "Enter a message: ";
	std::getline(std::cin, message);
	std::vector<double> input(vocabulary.size());
	std::vector<std::string> tokens = tokenize(message);

	for (const std::string &token: tokens)
	{
		for (int i = 0; i < (int) vocabulary.size(); ++i)
			if (token == vocabulary[i])
			{
				input[i] = 1;
				break;
			}
	}

	std::vector<double> output = nn.forward(input);
	std::vector<std::pair<double, int>> sortedOutput;
	sortedOutput.reserve(output.size());

	for (int i = 0; i < (int) output.size(); ++i) sortedOutput.emplace_back(output[i], i);
	std::sort(sortedOutput.begin(), sortedOutput.end(), std::greater<>());

	std::cout << "Possible responses: ";
	for (int i = 0; i < MAX_RESPONSES; ++i)
	{
		std::string response;
		for (int j = 0; j < (int) tokens.size(); ++j)
		{
			if (j > 0) response += ' ';
			response += vocabulary[sortedOutput[i].second];
		}

		std::cout << response << " (" << sortedOutput[i].first * 100 << "%) ";
	}

	std::cout << std::endl;
	return EXIT_SUCCESS;
}
