#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <filesystem>
#include <fstream>

#include "SudokuSolver/globals.h"
#include "SudokuSolver/dbn.h"

static const std::string image_name = "9image1-2.png";
static const bool save = 1;
static const bool load = 1;

void saveToFile(std::vector< std::vector< std::vector<float> > > & weights, std::string file);
void saveBiases(std::vector< std::vector<float> > & biases, std::string file);
void loadWeights(std::vector< std::vector< std::vector<float> > > & weights, std::string file);
void loadBiases(std::vector< std::vector<float> > & biases, std::string file);

int main( int argc, char** argv )
{
	DBN dbn;
    cv::Mat image;
	std::vector<cv::Mat> digits;
	std::vector<std::string> inputs;
	
	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
	std::tr2::sys::directory_iterator start(assets_dir + inputs_small_dir), end;
	for ( ; start != end; ++start) {
		inputs.push_back(start->path().string());
	}
	std::random_shuffle(inputs.begin(), inputs.end());
	
	if (load) {
		loadWeights(dbn.getWeights(), assets_dir + weights_file);
		loadBiases(dbn.getBiases(), assets_dir + biases_file);
	} else {
		for (size_t i = 0; i < inputs.size(); ++i) {
			std::cout << "#" << i << " " << inputs[i] << '\n';
			image = cv::imread(assets_dir + inputs_dir + inputs[i], cv::IMREAD_GRAYSCALE);
			dbn.setInput(image);
			dbn.train();
		}
		if (save) {
			saveToFile(dbn.getWeights(), assets_dir + weights_file);
			saveBiases(dbn.getBiases(), assets_dir + biases_file);
		}
	}

	image = cv::imread(assets_dir + inputs_dir + image_name, cv::IMREAD_GRAYSCALE);
    cv::imshow( image_window_name, image );

	// TODO test
	dbn.classify(image);
	cv::namedWindow( "test", cv::WINDOW_AUTOSIZE );
	cv::imshow("test", dbn.test_image);

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