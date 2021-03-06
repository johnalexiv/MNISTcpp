#include <iostream> 
#include <fstream>
#include <sstream> 
#include <vector>
#include "MNIST.h"
#include "NeuralNetwork.h"
using namespace std;

int main()
{
	bool train = false;

	if (train) {
		string imageFilename = "train-images.idx3-ubyte";
		string labelFilename = "train-labels.idx1-ubyte";
		NeuralNetwork model(imageFilename, labelFilename, 100, 0.01);

		int numOfEpochs = 10;
		model.train(numOfEpochs);
	}
	else {
		string imageFilename = "t10k-images.idx3-ubyte";
		string labelFilename = "t10k-labels.idx1-ubyte";
		NeuralNetwork model(100);
		MNIST mnist(imageFilename, labelFilename);

		model.loadWeights("weights.txt");

		auto numImages = mnist.getNumberOfImages();
		int correct = 0;
		int total = 0;
		for (int i = 0; i < 10; i++) {
			auto image = mnist.getImage2D();
			auto label = mnist.getLabel();
			auto index = model.prediction(image, label, true);
			if (label[index] == 1)
				correct++;
			total++;

			if (i % 100 == 0)
				cout << "Accuracy: " << (double)correct / total * 100.0 << endl;
		}
		cout << "Final accuracy: " << (double)correct / total * 100.0 << endl;
	}

	while (true) {}

	return 0;
}