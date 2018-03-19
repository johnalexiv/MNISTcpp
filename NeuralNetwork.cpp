#include "NeuralNetwork.h"

/*  Returns the product of two vectors (elementwise multiplication).
	Inputs:
		m1: vector
		m2: vector
	Output: vector, m1 * m2, product of two vectors m1 and m2
*/
vector<double> operator*(const vector<double> &m1, const vector<double> &m2) {
	const int size = m1.size();
	vector<double> output(size);

	for (int i = 0; i < size; i++)
		output[i] = m1[i] * m2[i];

	return output;
}

/*  Returns the product learning rate and vector (elementwise multiplication).
	Inputs:
		lr: double learning rate
		m: vector
	Output: vector, learning rate * each element in m
*/
vector<double> operator*(const double lr, const vector<double> &m) {
	const int size = m.size();
	vector<double> product(size);

	for (int i = 0; i < size; i++)
		product[i] = lr * m[i];

	return product;
}

/*  Returns the difference between two vectors.
Inputs:
m1: vector
m2: vector
Output: vector, m1 - m2, difference between two vectors m1 and m2.
*/
vector<double> operator-(const vector<double> &m1, const vector<double> &m2) {
	const int size = m1.size();
	vector<double> output(size);

	for (int i = 0; i < size; i++)
		output[i] = m1[i] - m2[i];

	return output;
}

/*  Returns the elementwise sum of two vectors.
Inputs:
m1: a vector
m2: a vector
Output: a vector, sum of the vectors m1 and m2.
*/
vector<double> operator+(const vector<double> &m1, const vector<double> &m2) {
	const int size = m1.size();
	vector<double> output(size);

	for (int i = 0; i < size; i++)
		output[i] = m1[i] + m2[i];

	return output;
} 

NeuralNetwork::NeuralNetwork(const int n) {
	numHiddenLayerNeurons = n;
	numOutputLayerNeurons = 10;
}

NeuralNetwork::NeuralNetwork(const string imageFilename, const string labelFilename, const int n, const double lr) {
	mnist = new MNIST(imageFilename, labelFilename);
	numHiddenLayerNeurons = n;
	numOutputLayerNeurons = 10;
	learningRate = lr;

	hiddenWeights.resize(numHiddenLayerNeurons * 784);
	outputWeights.resize(numHiddenLayerNeurons * numOutputLayerNeurons);

	hiddenBias.resize(numHiddenLayerNeurons);
	outputBias.resize(numOutputLayerNeurons);

	initializeWeights();
}

NeuralNetwork::~NeuralNetwork() {

}

void NeuralNetwork::train(const int numberEpochs) {
	int numberOfImages = mnist->getNumberOfImages();
	int progress = 0;
	int total = 0;
	int correct = 0;

	for (int i = 0; i < numberEpochs; i++) {
		cout << "Starting Epoch: " << i << endl;
		cout << numberOfImages << endl;
		for (int j = 0; j < numberOfImages; j++) {
			auto image = mnist->getImage();
			auto label = mnist->getLabel();
			backPropagation(image, label);
			if (progress % 1000 == 0) {
				auto image = mnist->getImage();
				auto label = mnist->getLabel();
				if (label[prediction(image, label)] == 1)
					correct++;
				cout << "Accuracy: " << (double)correct / total * 100.0 << endl;
				total++;
			}
			progress++;
		}
	}

	saveWeights();
}

void NeuralNetwork::backPropagation(const vector<double> &image, const vector<double> &label) {
	const int imageSize = image.size();

	vector<double> y1 = dot(image, hiddenWeights, 1, imageSize, numHiddenLayerNeurons) + hiddenBias;
	vector<double> a1 = hyperbolicTan(y1);

	vector<double> y2 = dot(a1, outputWeights, 1, numHiddenLayerNeurons, numOutputLayerNeurons) + outputBias;
	vector<double> a2 = hyperbolicTan(y2);

	vector<double> a2Error = label - a2;
	vector<double> a2Delta = a2Error * hyperbolicTan_(a2);
	vector<double> W2Delta = dot(transpose(a1, 1, numHiddenLayerNeurons), a2Delta, numHiddenLayerNeurons, 1, numOutputLayerNeurons);

	vector<double> a1Error = dot(a2Delta, transpose(outputWeights, numOutputLayerNeurons, numHiddenLayerNeurons), 1, numOutputLayerNeurons, numHiddenLayerNeurons);
	vector<double> a1Delta = a1Error * hyperbolicTan_(a1);
	vector<double> W1Delta = dot(transpose(image, 1, imageSize), a1Delta, imageSize, 1, numHiddenLayerNeurons);

	outputWeights = outputWeights + (learningRate * W2Delta);
	outputBias = outputBias + (learningRate * a2Delta);

	hiddenWeights = hiddenWeights + (learningRate * W1Delta);	
	hiddenBias = hiddenBias + (learningRate * a1Delta);
}

void printImage(const vector<double> &image) {
	int rows = 28;
	int cols = 28;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = i * cols + j;
			if (image[index] > 0)
				cout << "  ";
			else
				cout << "0 ";
		}
		cout << endl;
	}
}

/*	Returns the output of feeding one image through the network.
	Inputs: image: vector containing the image pixels [1, 784]
	Output: vector, [1, Number of Classes] containing predictions
			from neural network.
*/
int NeuralNetwork::prediction(const vector<double> &image, const vector<double> &label, bool verbose) {
	vector<double> y1 = dot(image, hiddenWeights, 1, image.size(), numHiddenLayerNeurons);
	vector<double> a1 = hyperbolicTan(y1);

	vector<double> y2 = dot(a1, outputWeights, 1, numHiddenLayerNeurons, numOutputLayerNeurons);
	vector<double> output = hyperbolicTan(y2);

	vector<double> error = label - output;
	
	double sum = 0.0;
	for (int i = 0; i < label.size(); i++) {
		sum += pow((label[i] - output[i]), 2.0);
	}

	if (verbose) {
		cout << "image: " << endl;
		printImage(image);
		cout << "output: ";
		print(output, 1, 10);
		cout << "label: ";
		print(label, 1, 10);
		cout << "error: ";
		print(error, 1, 10);
		cout << "overall error: " << sum / 10.0 << endl;
	}

	double max = output[0];
	int index = 0;
	for (int i = 1; i < output.size(); i++) {
		if (output[i] > max) {
			max = output[i];
			index = i;
		}
	}

	if (verbose) {
		cout << "prediction: " << index << endl;
		cout << endl;
	}

	return index;
}

void NeuralNetwork::loadWeights(const string fileName) {
	ifstream file;
	file.open(fileName);

	if (file.is_open()) {
		cout << "Loading weights..." << endl;

		int numOfHiddenWeights;
		double temp;
		file >> numOfHiddenWeights;
		hiddenWeights.resize(numOfHiddenWeights);
		for (int i = 0; i < numOfHiddenWeights; i++) 
			file >> hiddenWeights[i];

		int numOfHiddenBias;
		file >> numOfHiddenBias;
		hiddenBias.resize(numOfHiddenBias);
		for (int i = 0; i < numOfHiddenBias; i++)
			file >> hiddenBias[i];

		int numOfOutputWeights;
		file >> numOfOutputWeights;
		outputWeights.resize(numOfOutputWeights);
		for (int i = 0; i < numOfOutputWeights; i++)
			file >> outputWeights[i];

		int numOfOutputBias;
		file >> numOfOutputBias;
		outputBias.resize(numOfOutputBias);
		for (int i = 0; i < numOfOutputBias; i++)
			file >> outputBias[i];

		cout << "Done loading weights.." << endl;
		file.close();
	}
	else {
		cout << "Could not load weights.." << endl;
	}
}

void NeuralNetwork::saveWeights() {
	ofstream file;
	file.open("weights.txt");
	
	if (file.is_open()) {
		cout << "Saving weights..." << endl;

		file << hiddenWeights.size() << endl;
		for (int i = 0; i < hiddenWeights.size(); i++)
			file << hiddenWeights[i] << endl;

		file << hiddenBias.size() << endl;
		for (int i = 0; i < hiddenBias.size(); i++)
			file << hiddenBias[i] << endl;

		file << outputWeights.size() << endl;
		for (int i = 0; i < outputWeights.size(); i++)
			file << outputWeights[i] << endl;

		file << outputBias.size() << endl;
		for (int i = 0; i < outputBias.size(); i++)
			file << outputBias[i] << endl;

		cout << "Done saving weights..." << endl;
		file.close();
	}
	else {
		cout << "Failed to save weights.." << endl;
	}
}

void NeuralNetwork::initializeWeights() {
	random_device rd;
	default_random_engine generator(rd());
	normal_distribution<double> distribution(0, 0.16);

	for (int i = 0; i < hiddenBias.size(); i++)
		hiddenBias[i] = distribution(generator);

	for (int i = 0; i < outputBias.size(); i++)
		outputBias[i] = distribution(generator);

	for (int i = 0; i < hiddenWeights.size(); i++)
		hiddenWeights[i] = distribution(generator);

	for (int i = 0; i < outputWeights.size(); i++)
		outputWeights[i] = distribution(generator);
}

/*  Returns the dot product of two matrices: m1 x m2.
	Inputs:
		m1: vector, left matrix of size m1_rows x m1_columns
		m2: vector, right matrix of size m1_columns x m2_columns
		(the number of rows in the right matrix must be equal
		to the number of the columns in the left one)
		m1_rows: int, number of rows in the left matrix m1
		m1_columns: int, number of columns in the left matrix m1
		m2_columns: int, number of columns in the right matrix m2
	Output: vector, m1 * m2, product of two vectors m1 and m2,
		a matrix of size m1_rows x m2_columns
*/
vector<double> NeuralNetwork::dot(const vector<double> &m1, const vector<double> &m2,
									const int m1_rows, const int m1_cols, const int m2_cols) {
	vector<double> output(m1_rows * m2_cols);

	for (int row = 0; row < m1_rows; row++) {
		for (int col = 0; col < m2_cols; col++) {
			int index = row * m2_cols + col;
			output[index] = 0.0;
			for (int k = 0; k < m1_cols; k++) {
				output[index] += m1[row * m1_cols + k] * m2[k * m2_cols + col];
			}
		}
	}

	return output;
}

/*  Returns a transpose matrix of input matrix.
	Inputs:
		m: vector, input matrix
		col: int, number of columns in the input matrix
		row: int, number of rows in the input matrix
	Output: vector, transpose matrix mT of input matrix m
*/
vector<double> NeuralNetwork::transpose(const vector<double> &m, const int rows, const int cols) {
	vector <double> output(rows*cols);
	
	for (int n = 0; n != rows * cols; n++) {
		int i = n / cols;
		int j = n % cols;
		output[n] = m[rows*j + i];
	}

	return output;
}

/*	Returns the value of the tanh function.
	Input: m1, vector.
	Output: tanh for every element of the input matrix m1
*/
vector<double> NeuralNetwork::hyperbolicTan(const vector<double> &m) {
	const int size = m.size();
	vector<double> output(size);

	for (int i = 0; i < size; i++)
		output[i] = tanh(m[i]);
	
	return output;
}

vector<double> NeuralNetwork::hyperbolicTan_(const vector<double> &m) {
	const int size = m.size();
	vector<double> output(size);

	for (int i = 0; i < size; i++) {
		output[i] = (double)(1.0 - pow(m[i], 2.0));
	}

	return output;
}

/*	Prints the input vector as rows x cols matrix.
	Inputs:
		m: vector, matrix of size rows x cols
		rows: int, number of rows in matrix
		cols: int, number of columns in matrix
*/
void NeuralNetwork::print(const vector<double> &m, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = i * cols + j;
			cout << m[index] << " ";
		}
	}
	cout << endl;
}