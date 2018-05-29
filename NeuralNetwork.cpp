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

/*  Returns the product learning rate and vector (elementwise multiplication).
Inputs:
lr: double learning rate
m: vector
Output: vector, learning rate * each element in m
*/
vector<vector<double>> operator*(const double lr, const vector<vector<double>> &m) {
	const int size = m.size();
	vector<vector<double>> product(size, vector<double>(size));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			product[i][j] = lr * m[i][j];

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

/*  Returns the elementwise sum of two vectors.
Inputs:
m1: a vector
m2: a vector
Output: a vector, sum of the vectors m1 and m2.
*/
vector<vector<double>> operator+(const vector<vector<double>> &m1, const vector<vector<double>> &m2) {
	const int size = m1.size();
	vector<vector<double>> output(size, vector<double>(size));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			output[i][j] = m1[i][j] + m2[i][j];

	return output;
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

/*	Returns the value of the tanh function.
Input: m1, vector.
Output: tanh for every element of the input matrix m1
*/
vector<vector<double>> NeuralNetwork::hyperbolicTan(const vector<vector<double>> &m) {
	const int size = m.size();
	vector<vector<double>> output(size, vector<double>(size));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			output[i][j] = tanh(m[i][j]);

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

vector<vector<double>> NeuralNetwork::hyperbolicTan_(const vector<vector<double>> &m) {
	const int size = m.size();
	vector<vector<double>> output(size, vector<double>(size));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			output[i][j] = (double)(1.0 - pow(m[i][j], 2.0));

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

	kernel.resize(numFeatureMaps, vector<vector<double>>(5, vector<double>(5)));
	convBias.resize(numFeatureMaps, vector<vector<double>>(24, vector<double>(24)));

	hiddenWeights.resize(numHiddenLayerNeurons * 144 * numFeatureMaps);
	outputWeights.resize(numHiddenLayerNeurons * numOutputLayerNeurons);

	hiddenBias.resize(numHiddenLayerNeurons);
	outputBias.resize(numOutputLayerNeurons);

	initializeWeights();
}

NeuralNetwork::~NeuralNetwork() {

}

vector<vector<double>> NeuralNetwork::conv(const vector<vector<double>> &image, int numFeatureMap) {
	
	vector<vector<double>> output(image.size() - kernel[0].size() + 1, vector<double>(image.size() - kernel[0].size() + 1));

	for (int i = 0; i < image.size() - kernel[numFeatureMap].size() + 1; i++) {
		for (int j = 0; j < image[i].size() - kernel[numFeatureMap].size() + 1; j++) {
			double sum = 0.0;
			for (int n = 0; n < kernel[numFeatureMap].size(); n++) {
				for (int m = 0; m < kernel[numFeatureMap][n].size(); m++) {
					sum += image[i + n][j + m] * kernel[numFeatureMap][n][m];
					
				}
			}
			output[i][j] = sum;
		}
	}

	return output;
}

int findMaxPosition(const double a, const double b, const double c, const double d) {
	int position = 0;
	double maxValue = a;
	if (b > a) {
		maxValue = b;
		position = 1;
	}
	if (c > maxValue) {
		maxValue = c;
		position = 2;
	}
	if (d > maxValue) {
		maxValue = d;
		position = 3;
	}
	return position;
}

vector<vector<double>> NeuralNetwork::maxPool(const vector<vector<double>> &image, vector<int> &pos) {

	vector<vector<double>> output(image.size() / 2, vector<double>(image.size() / 2));
	double maxValue;
	int k = 0;
	int l = 0;
	int position = 0;
	for (int i = 0; i < image.size(); i += 2) {
		for (int j = 0; j < image.size(); j += 2) {
			switch (findMaxPosition(image[i][j], image[i][j + 1], image[i + 1][j], image[i + 1][j + 1])) {
			case 0:
				maxValue = image[i][j];
				pos.push_back(i);
				pos.push_back(j);
				break;
			case 1:
				maxValue = image[i][j + 1];
				pos.push_back(i);
				pos.push_back(j + 1);
				break;
			case 2:
				maxValue = image[i + 1][j];
				pos.push_back(i + 1);
				pos.push_back(j);
				break;
			case 3:
				maxValue = image[i + 1][j + 1];
				pos.push_back(i + 1);
				pos.push_back(j + 1);
				break;
			}
			output[k][l++] = maxValue;
		}
		k++;
		l = 0;
	}
	
	return output;
}

vector<double> NeuralNetwork::flatten(const vector<vector<vector<double>>> &image) {
	vector<double> output(image.size() * image[0].size() * image[0][0].size());

	for (int k = 0; k < image.size(); k++){
		for (int i = 0; i < image[k].size(); i++) {
			for (int j = 0; j < image[k][i].size(); j++) {
				int index = j + image[k][i].size()*(i + (image.size()*k));
				output[index] = image[k][i][j];
			}
		}
	}
	return output;
}

//vector<vector<double>> NeuralNetwork::unflatten(const vector<double> &image) {
//	vector<vector<double>> output((int)sqrt(image.size()), vector<double>((int)sqrt(image.size())));
//	
//	for (int i = 0; i < image.size(); i++) {
//		int row = i / output.size();
//		int col = i % output.size();
//		output[row][col] = image[i];
//	}
//
//	return output;
//}

vector<vector<vector<double>>> NeuralNetwork::unflatten(const vector<double> &image) {
	vector<vector<vector<double>>> output(numFeatureMaps, vector<vector<double>>((int)sqrt(image.size()), vector<double>((int)sqrt(image.size()))));
	for (int i = 0; i < image.size(); i++) {
		int temp_index = i;
		int featuremap = temp_index / (12 * 12);
		temp_index -= (featuremap * 12 * 12);
		int row = temp_index / output.size();
		int col = temp_index % output.size();
		output[featuremap][row][col] = image[i];
	}

	return output;
}

vector<vector<double>> NeuralNetwork::kernelDelta(const vector<vector<double>> &image, const vector<vector<double>> &p1, const vector<int> &pos) {
	vector<vector<double>> output(kernel[0].size(), vector<double>(kernel[0].size()));

	//(kernel.size(), vector<double>(kernel.size()));
	for (int i = 0; i < pos.size(); i += 2) {
		int x_pos = pos[i];
		int y_pos = pos[i + 1];

		for (int j = 0; j < kernel[0].size(); j++)
		{
			for (int k = 0; k < kernel[0].size(); k++)
			{
				output[j][k] += (p1[x_pos / 2][y_pos / 2] * image[j + (x_pos)][k + (y_pos)]);
			}
		}
	}

	return output;
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
			auto image = mnist->getImage2D();
			auto label = mnist->getLabel();
			backPropagation(image, label);
			if (progress % 1000 == 0) {
				auto image = mnist->getImage2D();
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

void NeuralNetwork::backPropagation(const vector<vector<double>> &image, const vector<double> &label) {
	const int imageSize = image.size();

	vector<vector<vector<double>>> y1(numFeatureMaps);

	for (int i = 0; i < numFeatureMaps; i++)
		y1[i] = conv(image, i);

	vector<vector<vector<double>>> a1(numFeatureMaps);
	for (int i = 0; i < numFeatureMaps; i++)
		a1[i] = hyperbolicTan(y1[i]);

	vector<vector<int>> pos(numFeatureMaps);
	vector<vector<vector<double>>> m1(numFeatureMaps);
	for (int i = 0; i < numFeatureMaps; i++)
		m1[i] = maxPool(a1[i], pos[i]);

	vector<double> f1 = flatten(m1);
	
	vector<double> y2 = dot(f1, hiddenWeights, 1, f1.size(), numHiddenLayerNeurons) + hiddenBias;
	vector<double> a2 = hyperbolicTan(y2);

	vector<double> y3 = dot(a2, outputWeights, 1, numHiddenLayerNeurons, numOutputLayerNeurons) + outputBias;
	vector<double> a3 = hyperbolicTan(y3);

	vector<double> a3Error = label - a3;
	vector<double> a3Delta = a3Error * hyperbolicTan_(a3);
	vector<double> W3Delta = dot(transpose(a2, 1, numHiddenLayerNeurons), a3Delta, numHiddenLayerNeurons, 1, numOutputLayerNeurons);

	vector<double> a2Error = dot(a3Delta, transpose(outputWeights, numOutputLayerNeurons, numHiddenLayerNeurons), 1, numOutputLayerNeurons, numHiddenLayerNeurons);
	vector<double> a2Delta = a2Error * hyperbolicTan_(a2);
	vector<double> W2Delta = dot(transpose(f1, 1, f1.size()), a2Delta, f1.size(), 1, numHiddenLayerNeurons);

	vector<double> a1Error = dot(a2Delta, transpose(hiddenWeights, 144*numFeatureMaps, numHiddenLayerNeurons), 1, numHiddenLayerNeurons, 144*numFeatureMaps);
	vector<double> a1Delta = a1Error * hyperbolicTan_(f1);
	vector<vector<vector<double>>> uf1 = unflatten(a1Delta);

	vector<vector<vector<double>>> W1Delta(numFeatureMaps);
	for (int i = 0; i < numFeatureMaps; i++)
		W1Delta[i]= kernelDelta(image, uf1[i], pos[i]);


	outputWeights = outputWeights + (learningRate * W3Delta);
	outputBias = outputBias + (learningRate * a3Delta);

	hiddenWeights = hiddenWeights + (learningRate * W2Delta);
	hiddenBias = hiddenBias + (learningRate * a2Delta);
	for (int i = 0; i < numFeatureMaps; i++)
		kernel[i] = kernel[i] + (learningRate * W1Delta[i]);
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

void printImage(const vector<vector<double>> &image) {
	for (int i = 0; i < image.size(); i++) {
		for (int j = 0; j < image.size(); j++) {
			if (image[i][j] > 0)
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
	vector<double> y1 = dot(image, hiddenWeights, 1, image.size(), numHiddenLayerNeurons) + hiddenBias;
	vector<double> a1 = hyperbolicTan(y1);

	vector<double> y2 = dot(a1, outputWeights, 1, numHiddenLayerNeurons, numOutputLayerNeurons) + outputBias;
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

/*	Returns the output of feeding one image through the network.
Inputs: image: vector containing the image pixels [1, 784]
Output: vector, [1, Number of Classes] containing predictions
from neural network.
*/
int NeuralNetwork::prediction(const vector<vector<double>> &image, const vector<double> &label, bool verbose) {
	const int imageSize = image.size();

	vector<vector<vector<double>>> y1(numFeatureMaps);

	for (int i = 0; i < numFeatureMaps; i++)
		y1[i] = conv(image, i);

	vector<vector<vector<double>>> a1(numFeatureMaps);
	for (int i = 0; i < numFeatureMaps; i++)
		a1[i] = hyperbolicTan(y1[i]);

	vector<vector<int>> pos(numFeatureMaps);
	vector<vector<vector<double>>> m1(numFeatureMaps);
	for (int i = 0; i < numFeatureMaps; i++)
		m1[i] = maxPool(a1[i], pos[i]);

	vector<double> f1 = flatten(m1);

	vector<double> y2 = dot(f1, hiddenWeights, 1, f1.size(), numHiddenLayerNeurons) + hiddenBias;
	vector<double> a2 = hyperbolicTan(y2);

	vector<double> y3 = dot(a2, outputWeights, 1, numHiddenLayerNeurons, numOutputLayerNeurons) + outputBias;
	vector<double> output = hyperbolicTan(y3);

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

		int kernelSize;
		file >> kernelSize;
		kernel.resize(numFeatureMaps, vector<vector<double>>(kernelSize, vector<double>(kernelSize)));
		for (int k = 0; k < numFeatureMaps; k++)
			for (int i = 0; i < kernelSize; i++)
				for (int j = 0; j < kernelSize; j++)
					file >> kernel[k][i][j];

		int numOfHiddenWeights;
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

		file << kernel[0].size() << endl;
		for (int k = 0; k < numFeatureMaps; k++)
			for (int i = 0; i < kernel[0].size(); i++)
				for (int j = 0; j < kernel[0].size(); j++)
					file << kernel[k][i][j] << endl;

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

	for (int k = 0; k < numFeatureMaps; k++)
		for (int i = 0; i < kernel[0].size(); i++)
			for (int j = 0; j < kernel[0].size(); j++)
				kernel[k][i][j] = distribution(generator);
	for (int k = 0; k < numFeatureMaps; k++)
		for (int i = 0; i < convBias[0].size(); i++)
			for (int j = 0; j < convBias[0].size(); j++)
				convBias[k][i][j] = distribution(generator);

	for (int i = 0; i < hiddenBias.size(); i++)
		hiddenBias[i] = distribution(generator);

	for (int i = 0; i < outputBias.size(); i++)
		outputBias[i] = distribution(generator);

	for (int i = 0; i < hiddenWeights.size(); i++)
		hiddenWeights[i] = distribution(generator);

	for (int i = 0; i < outputWeights.size(); i++)
		outputWeights[i] = distribution(generator);
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
