#include <iostream>

#include "solver.h"
#include "globals.h"

Solver::Solver()
{
	for (int i = 0; i < 9; ++i) {
		domain.push_back(i + 1);
	}
}

bool Solver::backtracking(int * sudoku, int empty)
{
	bool complete = true;
	for (int i = empty; i < NUM_CELLS && complete; ++i) {
		if (sudoku[i] == 0) {
			complete = false;
			empty = i;
		}
	}
	if (complete) {
		return true;
	}

	for (int value : domain) {
		if (checkConstraints(sudoku, empty, value)) {
			sudoku[empty] = value;
			if (backtracking(sudoku, empty+1)) {
				return true;
			}
			sudoku[empty] = 0;
		}
	}
	return false;
}

bool Solver::checkConstraints(int * sudoku, int empty, int value)
{
	// rows & columns
	int row = empty / 9;
	int column = empty % 9;
	for (int i = 0; i < 9; ++i) {
		if (sudoku[9 * row + i] == value) {
			return false;
		}
		if (sudoku[column + 9 * i] == value) {
			return false;
		}
	}

	// 3x3's
	row /= 3;
	column /= 3;
	row *= 3;
	column *= 3;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (sudoku[9 * (row + i) + (column + j)] == value) {
				return false;
			}
		}
	}

	return true;
}

void Solver::solve(int * sudoku)
{
	backtracking(sudoku, 0);
}