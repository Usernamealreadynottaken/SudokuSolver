#include "dbn.h"

DBN::DBN() : learning_rate(LEARNING_RATE), starting_mean(STARTING_MEAN), starting_std(STARTING_STD),
	num_epochs(NUM_EPOCHS)
{
	srand(static_cast<unsigned int>(time(0)));
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(starting_mean, starting_std);

	for (int i = 0; i < NUM_LAYERS; ++i) {
		weights.push_back(std::vector< std::vector<float> >());
		layers.push_back(std::vector<uchar>());
		layers_prim.push_back(std::vector<uchar>());
		biases.push_back(std::vector<float>());
	}
	biases.push_back(std::vector<float>());

	for (size_t i = 0; i < NUM_LAYERS; ++i) {
		for (size_t v = 0; v < NUM_NEURONS[i]; ++v) {
			layers[i].push_back(0);
			layers_prim[i].push_back(0);
			// TODO generalize
			biases[1-i].push_back(0.0f);
			weights[i].push_back(std::vector<float>());
			for (size_t h = 0; h < NUM_NEURONS[i+1]; ++h) {
				weights[i][v].push_back(distribution(generator));
			}
		}
	}

	for (size_t i = 0; i < NUM_NEURONS[NUM_LAYERS]; ++i) {
		biases[NUM_LAYERS].push_back(0.0f);
	}

	positive_gradient.reserve(NUM_NEURONS[0] * NUM_NEURONS[1]);
	negative_gradient.reserve(NUM_NEURONS[0] * NUM_NEURONS[1]);
	for (size_t i = 0; i < NUM_NEURONS[0] * NUM_NEURONS[1]; ++i) {
		positive_gradient.push_back(0);
		negative_gradient.push_back(0);
	}
	outputs.reserve(NUM_NEURONS[NUM_LAYERS]);
	for (size_t i = 0; i < NUM_NEURONS[NUM_LAYERS]; ++i) {
		outputs.push_back(0.0f);
	}
}

void DBN::train()
{
	calculateH();

	// positive gradient
	for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
		for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
			positive_gradient[h * NUM_NEURONS[0] + v] = layers[0][v] * layers[1][h];
		}
	}

	calculateVPrim();

	calculateHPrim();

	// negative gradient
	for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
		for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
			negative_gradient[h * NUM_NEURONS[0] + v] = layers_prim[0][v] * layers_prim[1][h];
		}
	}

	// update weights
	for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
		for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
			weights[0][v][h] += learning_rate * (positive_gradient[h * NUM_NEURONS[0] + v] - negative_gradient[h * NUM_NEURONS[0] + v]);
		}
	}
	// biases, not sure if correct
	for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
		biases[1][v] += learning_rate * (layers[0][v] - layers_prim[0][v]);
	}
	for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
		biases[0][h] += learning_rate * (layers[1][h] - layers_prim[1][h]);
	}
}

void DBN::calculateH()
{
	for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
		float value = biases[0][h];
		for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
			value += weights[0][v][h] * layers[0][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		if (value > ((float) rand() / (RAND_MAX))) {
			layers[1][h] = 1;
		} else {
			layers[1][h] = 0;
		}
	}
}

void DBN::calculateVPrim()
{
	for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
		float value = biases[1][v];
		for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
			value += weights[0][v][h] * layers[1][h];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		if (value > ((float) rand() / (RAND_MAX))) {
			layers_prim[0][v] = 1;
		} else {
			layers_prim[0][v] = 0;
		}
	}
}

void DBN::calculateHPrim()
{
	for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
		float value = biases[0][h];
		for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
			value += weights[0][v][h] * layers_prim[0][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		if (value > ((float) rand() / (RAND_MAX))) {
			layers_prim[1][h] = 1;
		} else {
			layers_prim[1][h] = 0;
		}
	}
}

void DBN::trainOutputs(int desired)
{
	// change number to index
	--desired;

	float value;
	float values[9];
	float max = -99999.0f;
	int max_index = 0;

	calculateH();
	for (size_t h = 0; h < NUM_NEURONS[2]; ++h) {
		value = biases[2][h];
		for (size_t v = 0; v < NUM_NEURONS[1]; ++v) {
			value += weights[1][v][h] * layers[1][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		outputs[h] = value;

		if (h == desired) {
			value = 1.0f - outputs[h];
		} else {
			value = 0.0f - outputs[h];
			value *= 0.125f;
		}
		values[h] = value;
		if (outputs[h] > max) {
			max = outputs[h];
			max_index = h;
		}
	}
	
	if (desired != max_index) {
		for (size_t h = 0; h < NUM_NEURONS[2]; ++h) {
			for (size_t v = 0; v < NUM_NEURONS[1]; ++v) {
				weights[1][v][h] += learning_rate * values[h];
			}
			biases[2][h] += learning_rate * values[h];
		}
	}
}

void DBN::trainPerceptron(int desired)
{
	--desired;
	float value;
	for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
		value = biases[1][h];
		for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
			value += weights[0][v][h] * layers[0][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		outputs[h] = value;

		if (h == desired) {
			value = 1.0f - outputs[h];
		} else {
			value = 0.0f - outputs[h];
		}
		
		for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
			if (layers[0][v])
				weights[0][v][h] += learning_rate * value;
		}
		biases[1][h] += learning_rate * value;
	}
}

int DBN::classify(cv::Mat image)
{
	setInput(image);
	calculateH();
	float value;
	float max = 0.0f;
	int result = 0;
	for (size_t h = 0; h < NUM_NEURONS[2]; ++h) {
		value = biases[2][h];
		for (size_t v = 0; v < NUM_NEURONS[1]; ++v) {
			value += weights[1][v][h] * layers[1][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		outputs[h] = value;
		if (value > max) {
			max = value;
			result = h;
		}
	}

	// TODO classify
	calculateVPrim();
	for (size_t i = 0; i < NUM_NEURONS[0]; ++i) {
		test_image.at<uchar>( i / 32, i % 32) = layers_prim[0][i] * 255;
	}

	return result + 1;
}

int DBN::classifyByPerceptron(cv::Mat image)
{
	setInput(image);
	float value;
	float max = 0.0f;
	int result = 0;
	for (size_t h = 0; h < NUM_NEURONS[1]; ++h) {
		value = biases[1][h];
		for (size_t v = 0; v < NUM_NEURONS[0]; ++v) {
			value += weights[0][v][h] * layers[0][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		outputs[h] = value;
		
		if (value > max) {
			max = value;
			result = h;
		}
	}

	return result + 1;
}

void DBN::setInput(cv::Mat & image)
{
	int avg = 0;
	for (int w = 0; w < image.size().width; ++w) {
		for (int h = 0; h < image.size().height; ++h) {
			avg += image.at<uchar>(w, h);
		}
	}
	avg /= 1024;
	avg = int(avg * 1.3);

	for (int w = 0; w < image.size().width; ++w) {
		for (int h = 0; h < image.size().height; ++h) {
			if (image.at<uchar>(w, h) < avg) {
				layers[0][WIDTH * w + h] = 0;
			} else {
				layers[0][WIDTH * w + h] = 1;
			}
		}
	}

	// TODO test
	test_image = image;
}

std::vector< std::vector< std::vector<float> > > & DBN::getWeights()
{
	return weights;
}

std::vector< std::vector<float> > & DBN::getBiases()
{
	return biases;
}