#include <opencv2/core/core.hpp>

#include <iostream>
#include <random>
#include <ctime>

#include "globals.h"

class DBN
{
private:
	float learning_rate;
	float starting_mean;
	float starting_std;
	int num_epochs;
	std::vector< std::vector< std::vector<float> > > weights;
	std::vector< std::vector<uchar> > layers;
	std::vector< std::vector<uchar> > layers_prim;
	std::vector< float > outputs;
	std::vector< std::vector<float> > biases;
	std::vector< uchar > positive_gradient;
	std::vector< uchar > negative_gradient;
protected:
	void calculateH(int n);
	void calculateVPrim(int n);
	void calculateHPrim(int n);
public:
	DBN();
	~DBN() { }
	void setInput(cv::Mat & image);
	void train(int layer);
	void trainOutputs(int desired);
	int classify(cv::Mat & image);
	std::vector< std::vector< std::vector<float> > > & getWeights();
	std::vector< std::vector<float> > & getBiases();

	
	// TODO test
	cv::Mat test_image;
};