#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include "MNIST.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(const int n);
	NeuralNetwork(const string imageFilename, const string labelFilename, const int n, const double lr);
	~NeuralNetwork();

	int prediction(const vector<double> &image, const vector<double> &label, bool verbose=false);
	void loadWeights(const string fileName);
	void train(const int numberEpochs);

private:
	friend vector<double> operator*(const vector<double> &m1, const vector<double> &m2);
	friend vector<double> operator*(const double learningRate, const vector<double> &m);
	friend vector<double> operator-(const vector<double> &m1, const vector<double> &m2);
	friend vector<double> operator+(const vector<double> &m1, const vector<double> &m2);

private:
	void backPropagation(const vector<double> &image, const vector<double> &label);
	
	vector<double> dot(const vector<double> &m1, const vector<double> &m2,
		const int m1_rows, const int m1_cols, const int m2_cols);
	vector<double> transpose(const vector<double> &m, const int rows, const int cols);

	vector<double> hyperbolicTan(const vector<double> &m);
	vector<double> hyperbolicTan_(const vector<double> &m);

	void saveWeights();
	void initializeWeights();
	void print(const vector<double> &m, int rows, int cols);

	int numHiddenLayerNeurons;
	int numOutputLayerNeurons;
	double learningRate;

	MNIST * mnist;
	
	vector<double> hiddenWeights;
	vector<double> hiddenBias;
	vector<double> outputWeights;
	vector<double> outputBias;
};