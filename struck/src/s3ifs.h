#ifndef S3IFS_H
#define S3IFS_H

#include <sdm/lib/utils.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>

extern std::ofstream trainingLogFile;
using SpMatRd = Eigen::SparseMatrix<double, Eigen::RowMajor>;

class s3ifs
{
public:
	s3ifs(SpMatRd X, SpMatRd D);
	~s3ifs();

private:
	int log_yn = 1;
	
	double rbu, rbl, rau, ral;
	int nbs, nas, max_iter, chk_fre, scr_max_iter;
	double gam, tol;
	double alpha, beta;
	int task;

	void parse_command_line();
};

#endif
