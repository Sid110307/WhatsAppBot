#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>

#include "include/neuralNetwork.h"

#define LEARNING_RATE 0.1
#define MOMENTUM 0.9
#define HIDDEN_LAYER_SIZE 8
#define EPOCHS 10

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
		if (line.find("Messages and calls are end-to-end encrypted.") != std::string::npos) continue;
		if (line.find("Disappearing messages were turned off.") != std::string::npos) continue;
		if (line.find("This message was deleted.") != std::string::npos) continue;
		if (line.find("You deleted this message.") != std::string::npos) continue;
		if (line.find('<') != std::string::npos) continue;
		if (line.find('>') != std::string::npos) continue;
		if (line.find("M - ") == std::string::npos) continue;

		size_t timestampPosition = line.find("M - ") + 4;
		size_t contentPosition = line.find(": ", timestampPosition) + 2;

		std::string sender = line.substr(timestampPosition, contentPosition - timestampPosition - 2);
		std::string content = line.substr(contentPosition);

		messages.push_back({sender, content});
	}

	file.close();
	return messages;
}

std::vector<std::string> tokenize(const std::string &str)
{
	std::vector<std::string> tokens;
	std::string token;

	for (char c: str)
		if (c == ' ' || c == '.' || c == ',' || c == '!' || c == '?')
		{
			if (!token.empty())
			{
				tokens.push_back(token);
				token.clear();
			}
			if (c != ' ')
			{
				std::string punct(1, c);
				tokens.push_back(punct);
			}
		} else token += c;

	if (!token.empty()) tokens.push_back(token);
	return tokens;
}

std::vector<std::string> getVocabulary(const std::vector<Message> &messages)
{
	std::map<std::string, int> vocabulary;
	std::vector<std::string> tokens;

	for (const Message &message: messages)
	{
		if (message.sender != USER) continue;

		std::vector<std::string> messageTokens = tokenize(message.content);
		for (const std::string &token: messageTokens)
		{
			if (vocabulary.find(token) == vocabulary.end()) vocabulary[token] = 1;
			else ++vocabulary[token];
		}
	}

	tokens.reserve(vocabulary.size());
	for (const auto &pair: vocabulary) tokens.push_back(pair.first);

	return tokens;
}

std::vector<std::vector<double>> getInputs(const std::vector<Message> &messages, const std::vector<std::string> &vocab)
{
	std::vector<std::vector<double>> inputs;
	for (const Message &message: messages)
	{
		if (message.sender != USER) continue;

		std::vector<double> input(vocab.size(), 0.0);
		std::vector<std::string> messageTokens = tokenize(message.content);

		for (const std::string &token: messageTokens)
		{
			auto it = std::find(vocab.begin(), vocab.end(), token);
			if (it != vocab.end()) input[it - vocab.begin()] = 1.0;
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
		std::vector<double> output(vocab.size(), 0.0);
		std::vector<std::string> messageTokens = tokenize(message.content);

		for (const std::string &token: messageTokens)
		{
			auto it = std::find(vocab.begin(), vocab.end(), token);
			if (it != vocab.end()) output[it - vocab.begin()] = 1.0;
		}

		outputs.push_back(output);
	}

	return outputs;
}

std::string respond(const std::string &userInput, const std::vector<std::string> &vocabulary, NeuralNetwork &nn)
{
	std::vector<double> input(vocabulary.size(), 0.0);
	std::vector<std::string> tokens = tokenize(userInput);

	for (const std::string &token: tokens)
	{
		auto it = std::find(vocabulary.begin(), vocabulary.end(), token);
		if (it != vocabulary.end()) input[it - vocabulary.begin()] = 1.0;
	}

	std::vector<std::vector<double>> output = nn.forward(input);
	std::string response;

	for (int i = 0; i < output[0].size(); ++i) if (output[0][i] > 0.5) response += vocabulary[i] + " ";
	return response;
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

	std::vector<int> layers = {(int) inputs[0].size(), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, (int) outputs[0].size()};
	NeuralNetwork nn(layers);

	if (nn.loadModel("model.bin")) std::cout << "Model loaded." << std::endl;
	else
	{
		std::cout << "Model doesn't exist. Training new model..." << std::endl;
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

			std::cout << "Epoch " << i + 1 << ": " << error / (int) inputs.size() << std::endl;
		}

		if (nn.saveModel("model.bin")) std::cout << "Model saved." << std::endl;
		else std::cerr << "Failed to save model." << std::endl;
	}

	std::cout << "You are: " << USER << std::endl;
	while (true)
	{
		std::string message;
		std::cout << "Enter a message (type QUIT to quit): ";

		std::getline(std::cin, message);
		if (message == "QUIT") break;

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

		std::vector<std::vector<double>> output = nn.forward(input);
		std::cout << "Bot: " << respond(message, vocabulary, nn) << std::endl;
	}

	return EXIT_SUCCESS;
}
