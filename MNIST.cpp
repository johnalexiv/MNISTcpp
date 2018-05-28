#include "MNIST.h"

MNIST::MNIST() {
	currentImageIndex = 0;
	currentLabelIndex = 0;
}

MNIST::MNIST(const string imageFilename, const string labelFilename) {
	if (!readMnistData(imageFilename))
		exit(0);
	if (!readMnistLabel(labelFilename))
		exit(0);
	currentImageIndex = 0;
	currentLabelIndex = 0;
}

MNIST::~MNIST() {

}

bool MNIST::readMnistData(string fileName) {
	ifstream inputFile(fileName, ios::binary);
	unsigned char temp = 0;
	int progress = 0;

	if (inputFile.is_open()) {
		int magicNumber = 0, numberOfImages = 0, rows = 0, cols = 0;

		inputFile.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = binaryValue(magicNumber);
		inputFile.read((char*)&numberOfImages, sizeof(numberOfImages));
		numberOfImages = binaryValue(numberOfImages);
		inputFile.read((char*)&rows, sizeof(rows));
		rows = binaryValue(rows);
		inputFile.read((char*)&cols, sizeof(cols));
		cols = binaryValue(cols);

		numOfImages = numberOfImages;
		images.resize(numberOfImages * 784);

		// set this back to images.size() from 784
		for (int i = 0; i < images.size(); i++) {
			inputFile.read((char*)&temp, sizeof(temp));
			images[i] = (double)temp / 255.0;

			progress++;
			if (progress % 100000 == 0)
				cout << "Loading MNIST Image Dataset Progress: " << fixed << setprecision(2) << ((double)progress / images.size() * 100.0) << "\r";
		}
		cout << endl;
		cout << "Loading MNIST Image Dataset Done." << endl;
		inputFile.close();
		return true;
	}
	else {
		cout << "Error: Could not open: " + fileName << endl;
		return false;
	}
}

bool MNIST::readMnistLabel(string fileName) {
	ifstream inputFile(fileName, ios::binary);
	unsigned char temp = 0;
	int progress = 0;

	if (inputFile.is_open()) {
		int magicNumber = 0, numberOfLabels = 0;

		inputFile.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = binaryValue(magicNumber);
		inputFile.read((char*)&numberOfLabels, sizeof(numberOfLabels));
		numberOfLabels = binaryValue(numberOfLabels);

		numOfLabels = numberOfLabels;
		labels.resize(numberOfLabels);

		for (int i = 0; i < numberOfLabels; i++) {
			inputFile.read((char*)&temp, 1);
			labels[i] = (double)temp;

			progress++;
			if (progress % 100 == 0)
				cout << "Loading MNIST Label Dataset Progress: " << fixed << setprecision(2) << ((double)progress / labels.size() * 100.0) << "\r";
		}
		cout << endl;
		cout << "Loading MNIST Label Dataset Done." << endl;
		inputFile.close();
		return true;
	}
	else {
		cout << "Error: Could not open: " + fileName << endl;
		return false;
	}
}

vector<double> MNIST::getImage() {
	vector<double> image(784);
	int nextImageindex = currentImageIndex + 784;

	if (nextImageindex >= images.size())
		nextImageindex = 0;

	int counter = 0;
	for (int i = currentImageIndex; i < nextImageindex; i++)
		image[counter++] = images[i];

	currentImageIndex = nextImageindex;
	return image;
}

vector<vector<double>> MNIST::getImage2D() {
	vector<double> image(784);
	int nextImageindex = currentImageIndex + 784;

	if (nextImageindex >= images.size())
		nextImageindex = 0;

	int counter = 0;
	for (int i = currentImageIndex; i < nextImageindex; i++)
		image[counter++] = images[i];

	currentImageIndex = nextImageindex;

	vector<vector<double>> image2D(28, vector<double>(28));

	for (int i = 0; i < image.size(); i++) {
		int row = i / 28;
		int col = i % 28;
		image2D[row][col] = image[i];
	}

	return image2D;
}

vector<double> MNIST::getLabel() {
	vector<double> label(10);

	if (currentLabelIndex == numOfLabels)
		currentLabelIndex = 0;

	// should be nextlabelindex and not currentlabelindex?
	double value = labels[currentLabelIndex];
	currentLabelIndex++;

	label = createLabel((int)value);
	return label;
}

vector<double> MNIST::createLabel(const int index) {
	vector<double> label(10);
	label[index] = 1.0;
	return label;
}

int MNIST::binaryValue(int value) {
	unsigned char c1, c2, c3, c4;
	c1 = value & 255;
	c2 = (value >> 8) & 255;
	c3 = (value >> 16) & 255;
	c4 = (value >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int MNIST::getNumberOfImages() {
	return numOfImages;
}

int MNIST::width() {
	return 28;
}

int MNIST::height() {
	return 28;
}