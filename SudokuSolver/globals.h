#pragma once

#define NUM_CELLS 81

static const std::string image_window_name = "Original";
static const std::string assets_dir = "../assets/";
static const std::string puzzles_dir = "puzzles/";
static const std::string digits_dir = "digits/";
static const std::string training_dir = "training/";
static const std::string inputs_dir = "tagged/";
static const std::string inputs_small_dir = "tagged_small/";
static const std::string weights_file = "weights.txt";
static const std::string biases_file = "biases.txt";

static const int WIDTH = 32;
static const int NUM_LAYERS = 1;
static const unsigned int NUM_NEURONS[] = {1024, 9};

static const float LEARNING_RATE = 0.001f;
static const float STARTING_MEAN = 0.0f;
static const float STARTING_STD = 0.01f;
static const int NUM_EPOCHS = 5;

static const float sigmoid_steepness = 1.0f;

static const float e = 2.71828182845904523536f;