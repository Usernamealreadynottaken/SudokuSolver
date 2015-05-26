#include <vector>

class Solver
{
private:
	std::vector<int> domain;
protected:
	bool backtracking(int * sudoku, int empty);
	bool checkConstraints(int * sudoku, int empty, int value);
public:
	Solver();
	~Solver() { }
	void solve(int * sudoku);
};