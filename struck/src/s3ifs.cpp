#include "s3ifs.h"
 
s3ifs::s3ifs(SpMatRd X, SpMatRd D)
{
    s3ifs::parse_command_line();
	if (log_yn == 1)
	{
		std::cout << "rbu: " << rbu << std::endl;
	}

    //ssvm_sifs solver(input_fn, alpha, beta, gam, tol, max_iter, chk_fre, scr_max_iter);
}

s3ifs::~s3ifs()
{}

void s3ifs::parse_command_line() {
    // default parameters
    rbu = 1.0;
    rbl = 0.05;
    nbs = 10;
    rau = 1.0;
    ral = 0.01;
    nas = 100;
    max_iter = 10000;
    gam = 0.05;
    tol = 1e-9;
    chk_fre = 1;
    scr_max_iter = 0;
    alpha = 1e-3;
    beta = 1e-3;
    task = -1;
}

