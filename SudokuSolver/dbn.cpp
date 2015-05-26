#include "dbn.h"

DBN::DBN() : learning_rate(LEARNING_RATE), starting_mean(STARTING_MEAN), starting_std(STARTING_STD),
	num_epochs(NUM_EPOCHS)
{
	srand(static_cast<unsigned int>(time(0)));
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(starting_mean, starting_std);
	
	for (int i = 0; i < NUM_LAYERS - 1; ++i) {
		weights.push_back(std::vector< std::vector<float> >());
		layers.push_back(std::vector<uchar>());
		layers_prim.push_back(std::vector<uchar>());
		biases.push_back(std::vector<float>());
		biases.push_back(std::vector<float>());
		for (size_t v = 0; v < NUM_NEURONS[i]; ++v) {
			layers[i].push_back(0);
			layers_prim[i].push_back(0);
			if (i % 2 == 0) {
				biases[i + 1].push_back(0.0f);
			} else {
				biases[i - 1].push_back(0.0f);
				biases[i + 1].push_back(0.0f);
			}
			weights[i].push_back(std::vector<float>());
			for (size_t h = 0; h < NUM_NEURONS[i+1]; ++h) {
				weights[i][v].push_back(distribution(generator));
			}
		}
	}
	
	outputs.reserve(NUM_NEURONS[NUM_LAYERS - 1]);
	for (size_t i = 0; i < NUM_NEURONS[NUM_LAYERS - 1]; ++i) {
		biases[NUM_LAYERS].push_back(0.0f);
		outputs.push_back(0.0f);
	}

	positive_gradient.reserve(NUM_NEURONS[0] * NUM_NEURONS[1]);
	negative_gradient.reserve(NUM_NEURONS[0] * NUM_NEURONS[1]);
	for (size_t i = 0; i < NUM_NEURONS[0] * NUM_NEURONS[1]; ++i) {
		positive_gradient.push_back(0);
		negative_gradient.push_back(0);
	}
}

void DBN::train(int layer)
{
	for (int i = 1; i <= NUM_LAYERS - 2; ++i) {
		calculateH(i);
	}
	// positive gradient
	for (size_t v = 0; v < NUM_NEURONS[layer-1]; ++v) {
		for (size_t h = 0; h < NUM_NEURONS[layer]; ++h) {
			positive_gradient[h * NUM_NEURONS[layer-1] + v] = layers[layer-1][v] * layers[layer][h];
		}
	}
	calculateVPrim(layer);

	calculateHPrim(layer);

	// negative gradient
	for (size_t v = 0; v < NUM_NEURONS[layer-1]; ++v) {
		for (size_t h = 0; h < NUM_NEURONS[layer]; ++h) {
			negative_gradient[h * NUM_NEURONS[layer-1] + v] = layers_prim[layer-1][v] * layers_prim[layer][h];
		}
	}
	
	// update weights
	for (size_t v = 0; v < NUM_NEURONS[layer-1]; ++v) {
		for (size_t h = 0; h < NUM_NEURONS[layer]; ++h) {
			weights[layer-1][v][h] += learning_rate * (positive_gradient[h * NUM_NEURONS[layer-1] + v] - negative_gradient[h * NUM_NEURONS[layer-1] + v]);
		}
	}
	// update biases
	for (size_t v = 0; v < NUM_NEURONS[layer-1]; ++v) {
		biases[layer/2+1][v] += learning_rate * (layers[layer-1][v] - layers_prim[layer-1][v]);
	}
	for (size_t h = 0; h < NUM_NEURONS[layer]; ++h) {
		biases[layer/2][h] += learning_rate * (layers[layer][h] - layers_prim[layer][h]);
	}
}

void DBN::calculateH(int n)
{
	for (size_t h = 0; h < NUM_NEURONS[n]; ++h) {
		float value = biases[n/2][h];
		for (size_t v = 0; v < NUM_NEURONS[n-1]; ++v) {
			value += weights[n-1][v][h] * layers[n-1][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		if (value > ((float) rand() / (RAND_MAX))) {
			layers[n][h] = 1;
		} else {
			layers[n][h] = 0;
		}
	}
}

void DBN::calculateVPrim(int n)
{
	for (size_t v = 0; v < NUM_NEURONS[n-1]; ++v) {
		float value = biases[n/2+1][v];
		for (size_t h = 0; h < NUM_NEURONS[n]; ++h) {
			value += weights[n-1][v][h] * layers[n][h];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		if (value > ((float) rand() / (RAND_MAX))) {
			layers_prim[n-1][v] = 1;
		} else {
			layers_prim[n-1][v] = 0;
		}
	}
}

void DBN::calculateHPrim(int n)
{
	for (size_t h = 0; h < NUM_NEURONS[n]; ++h) {
		float value = biases[n/2][h];
		for (size_t v = 0; v < NUM_NEURONS[n-1]; ++v) {
			value += weights[n-1][v][h] * layers_prim[n-1][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		if (value > ((float) rand() / (RAND_MAX))) {
			layers_prim[n][h] = 1;
		} else {
			layers_prim[n][h] = 0;
		}
	}
}

void DBN::trainOutputs(int desired)
{
	--desired;
	for (int i = 1; i <= NUM_LAYERS - 2; ++i) {
		calculateH(i);
	}
	float value;
	for (size_t h = 0; h < NUM_NEURONS[NUM_LAYERS - 1]; ++h) {
		value = biases[NUM_LAYERS][h];
		for (size_t v = 0; v < NUM_NEURONS[NUM_LAYERS - 2]; ++v) {
			value += weights[NUM_LAYERS - 2][v][h] * layers[NUM_LAYERS - 2][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		outputs[h] = value;

		if (h == desired) {
			value = 1.0f - outputs[h];
		} else {
			value = 0.0f - outputs[h];
		}
		
		for (size_t v = 0; v < NUM_NEURONS[NUM_LAYERS - 2]; ++v) {
			if (layers[NUM_LAYERS - 2][v]) {
				weights[NUM_LAYERS - 2][v][h] += learning_rate * value;
			}
		}
		biases[NUM_LAYERS][h] += learning_rate * value;
	}
}

int DBN::classify(cv::Mat & image)
{
	setInput(image);
	for (int i = 1; i <= NUM_LAYERS - 2; ++i) {
		calculateH(i);
	}
	float value;
	float max = 0.0f;
	int result = 0;
	for (size_t h = 0; h < NUM_NEURONS[NUM_LAYERS - 1]; ++h) {
		value = biases[NUM_LAYERS][h];
		for (size_t v = 0; v < NUM_NEURONS[NUM_LAYERS - 2]; ++v) {
			value += weights[NUM_LAYERS - 2][v][h] * layers[NUM_LAYERS - 2][v];
		}
		value = 1 / (1 + pow(e, -sigmoid_steepness * value));
		outputs[h] = value;
		if (value > max) {
			max = value;
			result = h;
		}
	}

	// TODO remove
	/*calculateVPrim(NUM_LAYERS - 2);
	int correct = 0;
	for (int i = 0; i < NUM_NEURONS[NUM_LAYERS - 3]; ++i) {
		if (layers_prim[NUM_LAYERS - 3][i] == layers[NUM_LAYERS - 3][i]) {
			++correct;
		}
	}
	std::cout << correct << '\n';*/
	/*for (int w = 0; w < image.size().width; ++w) {
		for (int h = 0; h < image.size().height; ++h) {
			test_image.at<uchar>(w, h) = 255 * layers_prim[0][WIDTH * w + h];
		}
	}*/

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

	// TODO remove
	//test_image = image;
}

std::vector< std::vector< std::vector<float> > > & DBN::getWeights()
{
	return weights;
}

std::vector< std::vector<float> > & DBN::getBiases()
{
	return biases;
}