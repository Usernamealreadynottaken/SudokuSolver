#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <filesystem>
#include <fstream>

#include "SudokuSolver/globals.h"
#include "SudokuSolver/dbn.h"

#include <ctime>

static const std::string image_name = "9image191-18.png";
static const bool save = 1;
static const bool load = 1;

void saveToFile(std::vector< std::vector< std::vector<float> > > & weights, std::string file);
void saveBiases(std::vector< std::vector<float> > & biases, std::string file);
void loadWeights(std::vector< std::vector< std::vector<float> > > & weights, std::string file);
void loadBiases(std::vector< std::vector<float> > & biases, std::string file);

int main( int argc, char** argv )
{
	DBN dbn;
	int result;
    cv::Mat image;
	std::vector<cv::Mat> digits;
	std::vector<std::string> inputs;
	
	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
	std::tr2::sys::directory_iterator start(assets_dir + inputs_dir), end;
	for ( ; start != end; ++start) {
		inputs.push_back(start->path().string());
	}
	
	if (load) {
		loadWeights(dbn.getWeights(), assets_dir + weights_file);
		loadBiases(dbn.getBiases(), assets_dir + biases_file);
		std::cout << "Loaded!\n";
	}
	for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
		std::random_shuffle(inputs.begin(), inputs.end());
		for (size_t i = 0; i < inputs.size(); ++i) {
			std::cout << "epoch " << epoch << " #" << i << " " << inputs[i] << '\n';
			image = cv::imread(assets_dir + inputs_dir + inputs[i], cv::IMREAD_GRAYSCALE);
			dbn.setInput(image);
			dbn.train(3);
			//dbn.trainOutputs(std::stoi(inputs[i].substr(0, 1)));
		}
	}
	
	/*bool done = false;
	int iteration = 0;
	int mistakes = 0;
	int index;
	while (!done) {
		mistakes = 0;
		done = true;
		std::cout << "iteration " << iteration++ << '\n';
		for (int i = 0; i < 50; ++i) {
			index = rand() % inputs.size();
			image = cv::imread(assets_dir + inputs_dir + inputs[index], cv::IMREAD_GRAYSCALE);
			dbn.setInput(image);
			dbn.trainOutputs(std::stoi(inputs[index].substr(0, 1)));
		}
		for (size_t i = 0; i < inputs.size(); ++i) {
			image = cv::imread(assets_dir + inputs_dir + inputs[i], cv::IMREAD_GRAYSCALE);
			if (std::stoi(inputs[i].substr(0, 1)) != dbn.classify(image)) {
				done = false;
				++mistakes;
			}
		}
		std::cout << mistakes << " mistakes\n";
	}*/

	clock_t start_clock = clock();
	clock_t end_clock = clock();

	int index;
	while (double(end_clock - start_clock) / CLOCKS_PER_SEC < 3600  * 5) {
		index = rand() % inputs.size();
		image = cv::imread(assets_dir + inputs_dir + inputs[index], cv::IMREAD_GRAYSCALE);
		dbn.setInput(image);
		dbn.trainOutputs(std::stoi(inputs[index].substr(0, 1)));
		end_clock = clock();
	}
	std::cout << double(end_clock - start_clock) / CLOCKS_PER_SEC << " second elapsed\n";

	// check how many mistakes
	int mistakes = 0;
	for (size_t i = 0; i < inputs.size(); ++i) {
		image = cv::imread(assets_dir + inputs_dir + inputs[i], cv::IMREAD_GRAYSCALE);
		if (std::stoi(inputs[i].substr(0, 1)) != dbn.classify(image)) {
			++mistakes;
		}
		if (i % 100 == 0) {
			std::cout << i << " checked\n";
		}
	}
	std::cout << mistakes << " mistakes total\n";

	if (save) {
		saveToFile(dbn.getWeights(), assets_dir + weights_file);
		saveBiases(dbn.getBiases(), assets_dir + biases_file);
		std::cout << "Saved!\n";
	}

	image = cv::imread(assets_dir + inputs_dir + image_name, cv::IMREAD_GRAYSCALE);
    cv::imshow( image_window_name, image );

	// TODO test
	//result = dbn.classify(image);
	//cv::imshow(image_window_name, dbn.test_image);
	//std::cout << "Classified as: " << result;

    cv::waitKey(0);
    return 0;
}

void saveToFile(std::vector< std::vector< std::vector<float> > > & weights, std::string file)
{
	std::vector< std::vector<float> > layer_weights;
	std::vector<float> neuron_weights;
	float weight;

	std::ofstream ofstream(file);
	if (ofstream.is_open()) {
		ofstream << weights.size() << '\n';
		for (size_t i = 0; i < weights.size(); ++i) {
			layer_weights = weights[i];
			ofstream << layer_weights.size() << '\n';
			for (size_t j = 0; j < layer_weights.size(); ++j) {
				neuron_weights = layer_weights[j];
				ofstream << neuron_weights.size() << '\n';
				for (size_t k = 0; k < neuron_weights.size(); ++k) {
					weight = neuron_weights[k];
					ofstream << weight << "\n";
				}
			}
		}
		ofstream.close();
	}
}

void saveBiases(std::vector< std::vector<float> > & biases, std::string file)
{
	std::vector<float> bias_values;
	float bias;

	std::ofstream ofstream(file);
	if (ofstream.is_open()) {
		ofstream << biases.size() << '\n';
		for (size_t i = 0; i < biases.size(); ++i) {
			bias_values = biases[i];
			ofstream << bias_values.size() << '\n';
			for (size_t j = 0; j < bias_values.size(); ++j) {
				bias = bias_values[j];
				ofstream << bias << '\n';
			}
		}
		ofstream.close();
	}
}

void loadWeights(std::vector< std::vector< std::vector<float> > > & weights, std::string file)
{
	size_t num_layers;
	size_t num_first_weights;
	size_t num_second_weights;

	std::ifstream ifstream(file);
	if (ifstream.is_open()) {
		ifstream >> num_layers;
		for (size_t i = 0; i < num_layers; ++i) {
			ifstream >> num_first_weights;
			for (size_t j = 0; j < num_first_weights; ++j) {
				ifstream >> num_second_weights;
				for (size_t k = 0; k < num_second_weights; ++k) {
					ifstream >> weights[i][j][k];
				}
			}
		}
		ifstream.close();
	}
}

void loadBiases(std::vector< std::vector<float> > & biases, std::string file)
{
	size_t num_layers;
	int num_values;

	std::ifstream ifstream(file);
	if (ifstream.is_open()) {
		ifstream >> num_layers;
		for (size_t i = 0; i < num_layers; ++i) {
			ifstream >> num_values;
			for (int j = 0; j < num_values; ++j) {
				ifstream >> biases[i][j];
			}
		}
		ifstream.close();
	}
}