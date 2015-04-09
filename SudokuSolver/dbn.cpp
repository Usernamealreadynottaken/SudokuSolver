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
	}
	layers.push_back(std::vector<uchar>());
	layers_prim.push_back(std::vector<uchar>());
	biases.push_back(std::vector<float>());

	for (int i = 0; i < NUM_INPUT_NEURONS; ++i) {
		layers[0].push_back(0);
		layers_prim[0].push_back(0);
		biases[1].push_back(0.0f);
	}

	for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
		weights[0].push_back(std::vector<float>());
		for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
			weights[0][v].push_back(distribution(generator));
		}
	}

	for (int i = 0; i < NUM_LAYER1_NEURONS; ++i) {
		layers[1].push_back(0);
		layers_prim[1].push_back(0);
		biases[0].push_back(0.0f);
	}

	positive_gradient.reserve(NUM_INPUT_NEURONS * NUM_LAYER1_NEURONS);
	negative_gradient.reserve(NUM_INPUT_NEURONS * NUM_LAYER1_NEURONS);
	for (size_t i = 0; i < NUM_INPUT_NEURONS * NUM_LAYER1_NEURONS; ++i) {
		positive_gradient.push_back(0);
		negative_gradient.push_back(0);
	}
}

void DBN::train()
{
	for (int i = 0; i < num_epochs; ++i) {

		calculateH();

		// positive gradient
		for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
			for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
				positive_gradient[h * NUM_INPUT_NEURONS + v] = layers[0][v] * layers[1][h];
			}
		}

		calculateVPrim();

		calculateHPrim();

		// negative gradient
		for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
			for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
				negative_gradient[h * NUM_INPUT_NEURONS + v] = layers_prim[0][v] * layers_prim[1][h];
			}
		}

		// update weights
		for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
			for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
				weights[0][v][h] += learning_rate * (positive_gradient[h * NUM_INPUT_NEURONS + v] - negative_gradient[h * NUM_INPUT_NEURONS + v]);
			}
		}
		// biases, not sure if correct
		for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
			biases[1][v] += learning_rate * (layers[0][v] - layers_prim[0][v]);
		}
		for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
			biases[0][h] += learning_rate * (layers[1][h] - layers_prim[1][h]);
		}

	}

}

void DBN::calculateH()
{
	for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
		float value = biases[0][h];
		for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
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
	for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
		float value = biases[1][v];
		for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
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
	for (int h = 0; h < NUM_LAYER1_NEURONS; ++h) {
		float value = biases[0][h];
		for (int v = 0; v < NUM_INPUT_NEURONS; ++v) {
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

void DBN::classify(cv::Mat image)
{
	setInput(image);
	calculateH();
	calculateVPrim();

	// TODO classify
	for (int i = 0; i < NUM_INPUT_NEURONS; ++i) {
		test_image.at<uchar>( i / 32, i % 32) = layers_prim[0][i] * 255;
	}
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