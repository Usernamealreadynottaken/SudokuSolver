#pragma once

#define NUM_ROWS 9
#define NUM_COLUMNS 9
#define NUM_CELLS 81

static const std::string image_window_name = "Original";
static const std::string assets_dir = "../assets/";
static const std::string puzzles_dir = "puzzles/";
static const std::string digits_dir = "test_digits/";
static const std::string training_dir = "training_puzzles/";
static const std::string test_dir = "test/";
static const std::string test_puzzles_dir = "test_puzzles/";
static const std::string inputs_dir = "training/";
static const std::string inputs_small_dir = "training_small/";
static const std::string weights_file = "weights.txt";
static const std::string biases_file = "biases.txt";

static const int WIDTH = 32;
static const int HEIGHT = 32;
static const unsigned int NUM_INPUTS = WIDTH * HEIGHT;
static const unsigned int NUM_OUTPUTS = 9;

// edit to change number of hidden layers
static const unsigned int NUM_NEURONS[] = {NUM_INPUTS, 500, NUM_OUTPUTS};

static const int NUM_LAYERS = sizeof(NUM_NEURONS) / sizeof(unsigned int);
static const float LEARNING_RATE = 0.001f;
static const float STARTING_MEAN = 0.0f;
static const float STARTING_STD = 0.01f;
static const int NUM_EPOCHS = 5;

static const float sigmoid_steepness = 1.0f;

static const float e = 2.71828182845904523536f;