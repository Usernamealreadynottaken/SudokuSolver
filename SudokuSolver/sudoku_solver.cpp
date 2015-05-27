#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

#include "globals.h"
#include "extractor.h"
#include "dbn.h"
#include "solver.h"

static std::string image_name = "simple1.jpg";

void loadWeights(std::vector< std::vector< std::vector<float> > > & weights, std::string file);
void loadBiases(std::vector< std::vector<float> > & biases, std::string file);

int main( int argc, char** argv )
{
	if (argc == 2) {
		image_name = argv[1];
	}

	int sudoku[NUM_CELLS];
	
	Extractor extractor;
    cv::Mat image;
	std::vector<cv::Mat> digits;
	DBN dbn;
	Solver solver;

	image = cv::imread(assets_dir + puzzles_dir + image_name, cv::IMREAD_GRAYSCALE);
	
	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( image_window_name, image );

	digits = extractor.extract(image, sudoku);

	loadWeights(dbn.getWeights(), assets_dir + weights_file);
	loadBiases(dbn.getBiases(), assets_dir + biases_file);
	std::cout << "Loaded!\n";

	int decision;
	int iterator = 0;
	for (int i = 0; i < NUM_COLUMNS; ++i) {
		for (int j = 0; j < NUM_ROWS; ++j) {

			if (sudoku[NUM_COLUMNS * i + j] == -1) {
				decision = dbn.classify(digits[iterator]);
				++iterator;
				sudoku[NUM_COLUMNS * i + j] = decision;
			}

		}
	}

	std::cout << "\nRecognized: \n";
	for (int i = 0; i < NUM_COLUMNS; ++i) {
		for (int j = 0; j < NUM_ROWS; ++j) {
			std::cout << sudoku[NUM_COLUMNS * i + j] << " ";
		}
		std::cout << '\n';
	}

	if (solver.solve(sudoku)) {
		std::cout << "\nSolved: \n";
		for (int i = 0; i < NUM_COLUMNS; ++i) {
			for (int j = 0; j < NUM_ROWS; ++j) {
				std::cout << sudoku[NUM_COLUMNS * i + j] << " ";
			}
			std::cout << '\n';
		}
	} else {
		std::cout << "\nCouldn't solve :(\n";
	}

	extractor.drawResults(image, sudoku);

    cv::waitKey(0);
    return 0;
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