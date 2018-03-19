#pragma once
#include <iostream> 
#include <fstream>
#include <sstream> 
#include <vector>
#include <iomanip>
using namespace std;

class MNIST
{
public:
	MNIST();
	MNIST(const string imageFilename, const string labelFilename);
	~MNIST();
	bool readMnistData(string);
	bool readMnistLabel(string);
	vector<double> getImage();
	vector<double> getLabel();
	int getNumberOfImages();
	int width();
	int height();
	
private:
	vector<double> createLabel(const int index);
	int binaryValue(int);

	int dataMagicNumber;
	int labelMagicNumber;
	int numOfImages;
	int numOfLabels;
	int numberOfRows;
	int numberOfCols;
	int currentImageIndex;
	int currentLabelIndex;
	vector<double> images;
	vector<double> labels;
};

