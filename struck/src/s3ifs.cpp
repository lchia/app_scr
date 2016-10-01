#include "s3ifs.h"

s3ifs::s3ifs(SpMatRd X)
{
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "*************************************"
						<< "*************************%%%%%%%%%%%%%Construction" 
						<< std::endl;
	}

    s3ifs::parse_command_line();
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "@@@@@@@@rbu: " << rbu << std::endl;
	}
	X_ = X;
    n_sams_ = X_.rows();
    n_feas_ = X_.cols();
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "@@@@@@@@n_sams_: " << n_sams_ << std::endl;
		trainingLogFile << "@@@@@@@@n_feas_: " << n_feas_ << std::endl;
	}
    X_CM_ = X_;

    inv_n_sams_ = 1.0 / static_cast<double>(n_sams_);
    inv_gamma_ = 1.0 / gamma_;
    inv_alpha_ = 1.0 / alpha_;
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "@@@@@@@@inv_n_sams_: " << inv_n_sams_ << std::endl;
		trainingLogFile << "@@@@@@@@inv_gamma_: " << inv_gamma_ << std::endl;
		trainingLogFile << "@@@@@@@@inv_alpha_: " << inv_alpha_ << std::endl;
	}
	
	//compute one_over_XTones_ & print
    one_over_XTones_ =
        inv_n_sams_ * (X_CM_.transpose() * Eigen::VectorXd::Ones(n_sams_)).array();

	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%%%%one_over_XTones_(" << n_feas_ << "):";
		for (int kk = 0; kk < n_sams_; kk++) 	
		{
			if (kk%15==0 || kk == n_sams_-1)
			{
				trainingLogFile << "(" << kk << ")" << one_over_XTones_[kk] << ",";
			}
		}
		trainingLogFile << std::endl;
	}

	//Compute beta_max_
    beta_max_ = one_over_XTones_.abs().maxCoeff();
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%beta_max_ = " << beta_max_ << std::endl;
	}
    if (beta_ >= beta_max_) {
        std::cout << "beta >= beta_max: always have naive solutions" << std::endl;
    }

	//FOR alpha_max_: temp = S_{beta_}(1/n * X_ * 1)  in Theorem 3
    Eigen::ArrayXd temp =
        (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
        (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%%%%temp(" << n_feas_ << "):";
		if (printMode == 1)
		{
			for (int kk = 0; kk < n_sams_; kk++) 	
			{
				if (kk%15==0 || kk == n_sams_-1) 
				{
					trainingLogFile << "(" << kk << ")" << temp[kk] << ",";
				}
			}
			trainingLogFile << std::endl;
		}
		else
		{
			trainingLogFile << temp[0] << "	...	" << temp[n_sams_-1] << std::endl;
		}
	}
 
	//Compute alpha_max_ in Theorem 3.
    alpha_max_ = (1.0/(1.0 - gamma_)) * (X_ * temp.matrix()).maxCoeff();
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%alpha_max_ = " << alpha_max_ << std::endl;
	}

    if (alpha_max_ > 0) {
        ref_alpha_ = alpha_max_;
        ref_dsol_ = Eigen::VectorXd::Ones(n_sams_);	//theta*
        ref_psol_ = temp * (1.0 / ref_alpha_);		// w*
        dif_alpha_ratio_ = 0.5 * (alpha_ - ref_alpha_) / alpha_;
        sum_alpha_ratio_ = 0.5 * (alpha_ + ref_alpha_) / alpha_;
    } else{
        ref_alpha_ = 0;
        ref_dsol_ = Eigen::VectorXd::Ones(n_sams_);
        ref_psol_ = Eigen::VectorXd::Zero(n_feas_); //Why? ->
        dif_alpha_ratio_ = 0.5;
        sum_alpha_ratio_ = 0.5;
    } 

    dsol_ = Eigen::VectorXd::Ones(n_sams_);	//Dual optimum
    Xw_comp_ = dsol_;
    psol_ = Eigen::VectorXd::Zero(n_feas_);	//Primal optimum
    XTdsol_ = Eigen::VectorXd::Zero(n_feas_);

    Xi_norm_.resize(n_sams_);
    for (int i = 0; i < n_sams_; ++i) {
        Xi_norm_[i] = X_.row(i).norm();
    }
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%Xi_norm_ = [" 
			<< Xi_norm_[0] << "	...	" << Xi_norm_[n_sams_-1] << "]" << std::endl;
	}

    Xj_norm_.resize(n_feas_);
    for (int j = 0; j < n_feas_; ++j) {
        Xj_norm_[j] = X_CM_.col(j).norm();
    } 
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%Xi_norm_ = [" 
			<< Xj_norm_[0] << "	...	" << Xj_norm_[n_feas_-1] << "]" << std::endl;
	}
	 
    Xi_norm_sq_ = Xi_norm_.square();
    Xj_norm_sq_ = Xj_norm_.square();
 
    all_ins_index_.resize(n_sams_);
    std::iota(std::begin(all_ins_index_), std::end(all_ins_index_), 0);
    all_fea_index_.resize(n_feas_);
    std::iota(std::begin(all_fea_index_), std::end(all_fea_index_), 0);

    idx_Fc_flag_ = Eigen::VectorXd::Ones(n_feas_);
    idx_Dc_flag_ = Eigen::VectorXd::Ones(n_sams_);

    idx_Dc_.clear();
    idx_Fc_.clear();
    idx_Dc_.insert(std::begin(all_ins_index_), std::end(all_ins_index_));
    idx_Fc_.insert(std::begin(all_fea_index_), std::end(all_fea_index_));

    idx_nsv_L_.clear();
    idx_nsv_R_.clear();
    idx_F_.clear();

    std::copy(std::begin(all_ins_index_), std::end(all_ins_index_),
              std::back_inserter(idx_Dc_vec_));
    std::copy(std::begin(all_fea_index_), std::end(all_fea_index_),
              std::back_inserter(idx_Fc_vec_));
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "%%%%%%%%%%%%%%idx_Dc_vec_ = [" 
			<< idx_Dc_vec_[0] << "	...	" << idx_Dc_vec_[n_sams_-1] << "]" << std::endl;
		trainingLogFile << "%%%%%%%%%%%%%%idx_Fc_vec_ = [" 
			<< idx_Fc_vec_[0] << "	...	" << idx_Fc_vec_[n_feas_-1] << "]" << std::endl;
	}

    approx_dsol_r_L_sq_ = 0.0;
    approx_dsol_r_R_sq_ = 0.0;
    approx_psol_r_F_sq_ = 0.0;
    
    fea_scr_time_ = 0.0;
    sam_scr_time_ = 0.0;
    scr_time_ = 0.0;

    iter_ = 0;
    duality_gap_ = std::numeric_limits<double>::max();
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "....................%s3ifs Construction%duality_gap_ = " 
			<< duality_gap_ << std::endl;
	}
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "*************************************"
						<< "*************************%%%%%%%%%%%%%Construction END" 
						<< std::endl << std::endl;
	}
    //s3ifs solver(input_fn, alpha, beta, gam, tol, max_iter, chk_fre, scr_max_iter);
}

s3ifs::~s3ifs()
{}

void s3ifs::printMat(SpMatRd M)
{
	trainingLogFile << std::endl;
	trainingLogFile << "Matrix: [" << M.rows() << "," << M.cols() << "]" << std::endl;
	int printMode = 0;
	if (printMode == 1)
	{
		for(int i = 0; i < M.rows(); i++)
		{
			Eigen::ArrayXd M_row = M.row(i);

			trainingLogFile << "Row[" << i << "]:";
			for(int j = 0; j < M.cols(); j++)
			{
				trainingLogFile << M_row[j] << " ";
			}
			trainingLogFile  << std::endl;
		}
	}
	else
	{	
		Eigen::ArrayXd M_row_1 = M.row(0);
		Eigen::ArrayXd M_row_2 = M.row(M.rows()-1); 
		trainingLogFile << "        [" 
						<< M_row_1[0] << " ... " << M_row_1[M.cols()-1] << "]"
						<< std::endl;
		trainingLogFile << "        "  << "	.				." << std::endl
						<< "        "  << "	.				." << std::endl;
		trainingLogFile << "        ["   
						<< M_row_2[0] << " ... " << M_row_2[M.cols()-1] << "]"
						<< std::endl;
	}
	trainingLogFile  << std::endl;
}
int s3ifs::get_n_sams(void) const { return n_sams_; }
int s3ifs::get_n_feas(void) const { return n_feas_; }

double s3ifs::get_primal_obj(void) const { return pobj_; }
double s3ifs::get_dual_obj(void) const { return dobj_; }
double s3ifs::get_duality_gap(void) const { return duality_gap_; }

int s3ifs::get_n_L(void) const { return idx_nsv_L_.size(); }
int s3ifs::get_n_R(void) const { return idx_nsv_R_.size(); }
int s3ifs::get_n_F(void) const { return idx_F_.size(); }

int s3ifs::get_iter(void) const { return iter_; }

double s3ifs::get_alpha_max(void) const { return alpha_max_; }
double s3ifs::get_beta_max(void) const { return beta_max_; }

Eigen::VectorXd s3ifs::get_dual_sol(void) const { return dsol_; }
Eigen::VectorXd s3ifs::get_primal_sol(void) const { return psol_; }

void s3ifs::set_stop_tol(const double& tol) { tol_ = tol; }

void s3ifs::compute_primal_obj(const bool& flag_comp_loss) 
{
    pobj_ = (.5 * alpha_) * psol_.squaredNorm() + (beta_) * psol_.lpNorm<1>();

    loss_ = 0.0;
    if (flag_comp_loss)
	{
        Xw_comp_ = 1 - (X_ * psol_).array();
        double Xw_comp_i;
        for (int i = 0; i < n_sams_; ++i) 
		{
            Xw_comp_i = Xw_comp_[i];
            if (Xw_comp_i > gamma_)
			{
                loss_ += Xw_comp_i - .5 * gamma_;
            } 
			else if (Xw_comp_i > 0.0) 
			{
                loss_ += .5 * Xw_comp_i * Xw_comp_i * inv_gamma_;
            }
        }
    }
    pobj_ = pobj_ + loss_ * inv_n_sams_;
}

void s3ifs::compute_dual_obj(const bool& flag_comp_XTdsol) 
{
    if (flag_comp_XTdsol)
	{
        update_psol(true);
    }
    Eigen::ArrayXd temp = XTdsol_.array().abs() - beta_;
    dobj_ =
        (dsol_.sum() - (.5 * gamma_) * dsol_.squaredNorm()) * inv_n_sams_
        - (0.5 * inv_alpha_) * (temp >= 0.0).select(temp, 0.0).square().sum();
}

void s3ifs::compute_duality_gap(const bool& flag_comp_loss, const bool& flag_comp_XTdsol) 
{
    compute_primal_obj(flag_comp_loss);
    compute_dual_obj(flag_comp_XTdsol);

    duality_gap_ = std::max(0.0, pobj_ - dobj_);
}

void s3ifs::update_psol(const bool& flag_comp_XTdsol) {
    if (flag_comp_XTdsol) {
        XTdsol_.setZero();
        for (int i = 0; i < n_sams_; ++i) {
            if (dsol_[i] > 0.0) {
                XTdsol_ += dsol_[i] * X_.row(i);
            }
        }
        XTdsol_ *= inv_n_sams_;
    }

    for (int i = 0; i < n_feas_; ++i) {
        psol_[i] = val_sign(XTdsol_[i]) * inv_alpha_ *
            std::max(0.0, std::abs(XTdsol_[i]) - beta_);
    }
}

void s3ifs::train_sifs(const int& scr_option) 
{
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "=================================s3ifs::train_sifs "
						<< std::endl;
	}
    // we have closed form solution, if alpha > alpha_max(beta)
    if (alpha_ >= alpha_max_) {
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << "...........#train_sifs#<alpha_, alpha_max_>=<" 
				<< alpha_ << ", " << alpha_max_ 
				<< ">: We have closed form solution, if alpha > alpha_max(beta)." 
				<< std::endl;
		}
        dsol_.setOnes();
        Eigen::ArrayXd temp =
            (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
            (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
        psol_ = temp * inv_alpha_;
        duality_gap_ = 0.0;
        return;
    }

    if (scr_option == 0) {        
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << ".............#train_sifs#scr_option=" << scr_option
				<< "-> alternative safe screening, and sample screening first." 
				<< std::endl;
		}
        sifs(true);
    } else if (scr_option == 1) {  
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << ".............#train_sifs#scr_option=" << scr_option
				<< "-> alternative safe screening, and feature screening first." 
				<< std::endl;
		}
        sifs(false);
    } else if (scr_option == 2) {
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << ".............#train_sifs#scr_option=" << scr_option
				<< "-> only sample screening." 
				<< std::endl;
		}
        iss();
    } else if (scr_option == 3) {
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << ".............#train_sifs#scr_option=" << scr_option
				<< "-> only feature screening." 
				<< std::endl;
		}
        ifs();
    }

    //XTdsol_ =  X*dsol_/n, used in (KKT-1)
    XTdsol_.setZero();
    for (int i = 0; i < n_sams_; i++) {
        if (dsol_[i] > 0.0) {
            XTdsol_ += dsol_[i] * X_.row(i);
        }
    }
    XTdsol_ *= inv_n_sams_;
    for (auto &&kk : idx_Fc_vec_) 
	{
        psol_[kk] = inv_alpha_ * val_sign(XTdsol_[kk]) *
            std::max(0.0, std::abs(XTdsol_[kk]) - beta_);
    }
    compute_duality_gap(true, false);
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "....................#train_sifs#duality_gap_ = " 
			<< duality_gap_ << std::endl;
	}

    int n_Dc = idx_Dc_vec_.size();
    if (n_Dc == 0) {
        std::cout << "  All samples screened. return" << std::endl;
        return;
    }

    int ind = 0;
    const double inv_nalpha_ = inv_n_sams_ * inv_alpha_;
    double delta_ind = 0.0;
    double p_theta_ind = 0.0;

    std::default_random_engine rg;
    std::uniform_int_distribution<> uni_dist(0, n_Dc - 1);

    const auto ins_begin_it = std::begin(idx_Dc_vec_);//
    auto random_it = std::next(ins_begin_it, uni_dist(rg));

    for (iter_ = 1; iter_ < max_iter_ && duality_gap_ > tol_; ++iter_) 
	{
		if (!trainingLogFile && debugMode == 1)	{
			trainingLogFile << "............#train_sifs#iter_ = " 
				<< iter_ << "->(max_iter_) " << max_iter_ << std::endl;
		}
        for (int jj = 0; jj < n_Dc; ++jj) 
		{
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#jj = " 
					<< jj << "->(n_Dc) " << n_Dc << std::endl;
			}

            random_it = std::next(ins_begin_it, uni_dist(rg));
            ind = *random_it;
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#ind = " 
					<< ind << std::endl;
			}

            p_theta_ind = dsol_[ind];
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#p_theta_ind = " 
					<< p_theta_ind << std::endl;
			}

			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#Xi_norm_sq_[ind]"
					<< Xi_norm_sq_[ind] << std::endl;
				trainingLogFile << "................#train_sifs#(X_.row(ind) * psol_)"
					<< X_.row(ind) * psol_ << std::endl;
			}
            delta_ind = (1 - gamma_ * p_theta_ind - (X_.row(ind) * psol_)(0)) /
                (gamma_ + Xi_norm_sq_[ind] * inv_nalpha_);
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#delta_ind@1 = " 
					<< delta_ind << std::endl;
			}
            delta_ind = std::max(-p_theta_ind, std::min(1.0 - p_theta_ind,
                                                        delta_ind));
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#delta_ind@2 = " 
					<< delta_ind << std::endl;
			}
            dsol_[ind] += delta_ind; 
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#dsol_[ind] = " 
					<< dsol_[ind] << std::endl;
			}

            XTdsol_ +=  (delta_ind * inv_n_sams_) * X_.row(ind);
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#XTdsol_ = " 
					<< XTdsol_.size() << std::endl;
			}


            for (auto &&kk : idx_Fc_vec_) {
                psol_[kk] = inv_alpha_ * val_sign(XTdsol_[kk]) *
                    std::max(0.0, std::abs(XTdsol_[kk]) - beta_);
            }
			if (!trainingLogFile && debugMode == 1 && jj%50==0)	{
				trainingLogFile << "................#train_sifs#psol_ = " 
					<< psol_.size() << std::endl;
			}
        }

        if (iter_ % chk_fre_ == 0) {
            compute_duality_gap(true, false);
			if (trainingLogFile && debugMode == 1)	{
				trainingLogFile << "...............#train_sifs#Iter: " << iter_ 
								<< "; Primal obj: " << get_primal_obj()
                     			<< "; Dual obj: " << get_dual_obj()
                    			<< "; Duality gap: " << get_duality_gap() 
								<< std::endl;
			}
            //std::cout<< "    Iter: " << iter_ << " Primal obj: " << get_primal_obj()
            //         << " Dual obj: " << get_dual_obj()
            //         << " Duality gap: " << get_duality_gap() << std::endl;
        }
    }
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "=================================s3ifs::train_sifs  END!====="
						<< std::endl;
	}
}

void s3ifs::clear_idx(void)
{
    idx_Fc_vec_.clear();
    idx_Dc_vec_.clear();
    std::copy(std::begin(all_ins_index_), std::end(all_ins_index_),
              std::back_inserter(idx_Dc_vec_));
    std::copy(std::begin(all_fea_index_), std::end(all_fea_index_),
              std::back_inserter(idx_Fc_vec_));

    idx_Dc_.clear();
    idx_Fc_.clear();
    idx_Dc_.insert(std::begin(all_ins_index_), std::end(all_ins_index_));
    idx_Fc_.insert(std::begin(all_fea_index_), std::end(all_fea_index_));

    idx_nsv_L_.clear();
    idx_nsv_R_.clear();
    idx_F_.clear();

    approx_psol_r_F_sq_ = 0.0;
    approx_dsol_r_R_sq_ = 0.0;
    approx_dsol_r_L_sq_ = 0.0;
    approx_psol_r_sq_ = 0.0;
    approx_dsol_r_sq_ = 0.0;

    idx_Fc_flag_.setOnes();
    idx_Dc_flag_.setOnes();
}

void s3ifs::set_alpha(const double& alpha, const bool& ws) {
    if (ws) 
	{  // solved problem as warm start and reference solutions
        ref_alpha_ = alpha_;
        alpha_ = alpha;
        inv_alpha_ = 1.0 / alpha_;
        ref_psol_ = psol_;
        ref_dsol_ = dsol_;
    } 
	else 
	{  // no warm start, use naive solution as reference solutions
        alpha_ = alpha;
        inv_alpha_ = 1.0 / alpha_;
        ref_alpha_ = alpha_max_;
        Eigen::ArrayXd temp =
            (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
            (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
        ref_psol_ = temp * (1.0 / ref_alpha_);
        ref_dsol_.resize(n_sams_);
        ref_dsol_.setOnes();

        // set reference solutions as initial
        psol_ = ref_psol_;
        dsol_ = ref_dsol_;
    }

    dif_alpha_ratio_ = 0.5 * (alpha_ - ref_alpha_) / alpha_;
    sum_alpha_ratio_ = 0.5 * (alpha_ + ref_alpha_) / alpha_;
 
    duality_gap_ = std::numeric_limits<double>::max();
    iter_ = 0;
    fea_scr_time_ = 0.0;
    sam_scr_time_ = 0.0;
    scr_time_ = 0.0;

    clear_idx();
}

void s3ifs::set_beta(const double& beta) {
    beta_ = beta;
    if (beta_ >= beta_max_) 
	{
        std::cout << "beta >= beta_max: always naive solution" << std::endl;
    }
    Eigen::ArrayXd temp =
        (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
        (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
    alpha_max_ = (1.0/(1.0 - gamma_)) * (X_ * temp.matrix()).maxCoeff();
    if (alpha_max_ > 0) 
	{
        ref_alpha_ = alpha_max_;
        ref_dsol_.resize(n_sams_);
        ref_dsol_.setOnes();
        ref_psol_ = temp * (1.0 / ref_alpha_);

        dif_alpha_ratio_ = 0.5 * (alpha_ - ref_alpha_) / alpha_; //First coeff. of (3)
        sum_alpha_ratio_ = 0.5 * (alpha_ + ref_alpha_) / alpha_; //second coeff. of (3)
    } 
	else 
	{
        ref_alpha_ = 0;
        ref_dsol_ = Eigen::VectorXd::Ones(n_sams_);
        ref_psol_ = Eigen::VectorXd::Zero(n_feas_);

        dif_alpha_ratio_ = 0.5;
        sum_alpha_ratio_ = 0.5;
    }
    
    duality_gap_ = std::numeric_limits<double>::max();
    iter_ = 0;
    fea_scr_time_ = 0.0;
    sam_scr_time_ = 0.0;
    scr_time_ = 0.0;

    clear_idx();
}

void s3ifs::sample_screening(void) 
{
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "++++++++++++++++++++++++++++s3ifs::sample_screening"
						<< std::endl;
	}
    auto start_time = sys_clk::now();

    const double L_coeff = 0.5 * ((2 * gamma_ - 1) * alpha_ + ref_alpha_) *
        inv_alpha_ * inv_gamma_; //used in the second item of (6)

    double psol_radius =
        std::sqrt(std::max(0.0, dif_alpha_ratio_ * dif_alpha_ratio_ *
                           approx_psol_r_sq_ - sum_alpha_ratio_ *
                           sum_alpha_ratio_ * approx_psol_r_F_sq_));

    double temp, xiw_comp_lb, xiw_comp_ub, Xi_Fc_sq, Xi_Fc_approx_psol_Fc;
    std::vector<int> new_nsv_L, new_nsv_R;

    for (auto &&i : idx_Dc_) 
	{
        Xi_Fc_sq = 0.0;

        for (srm_iit it(X_, i); it; ++it) 
		{
            if (idx_Fc_flag_[it.index()])
                Xi_Fc_sq += it.value() * it.value(); //Used in (10)
        }

        Xi_Fc_approx_psol_Fc = sum_alpha_ratio_ * (X_.row(i) * ref_psol_)(0);
        temp = psol_radius * std::sqrt(Xi_Fc_sq); //See Lemma 5
        xiw_comp_ub = 1 - Xi_Fc_approx_psol_Fc + temp;
        xiw_comp_lb = 1 - Xi_Fc_approx_psol_Fc - temp;
 
        if (xiw_comp_ub <  - 1e-9) 
		{
            new_nsv_R.push_back(i);
        }
		else if (xiw_comp_lb > gamma_ + 1e-9) 
		{
            new_nsv_L.push_back(i);
        }
    }

    for (auto &&i : new_nsv_R) 
	{
        dsol_[i] = 0.0;
        idx_Dc_flag_[i] = 0.0;
        idx_nsv_R_.insert(i);
        idx_Dc_.erase(i);
        approx_dsol_c_[i] = 0.0;
        temp = dif_alpha_ratio_ * inv_gamma_ + sum_alpha_ratio_ * ref_dsol_[i]; //Third item of (6)
        approx_dsol_r_R_sq_ += temp * temp; //Third item of (6)
    }
    for (auto &&i : new_nsv_L) 
	{
        dsol_[i] = 1.0;
        idx_Dc_flag_[i] = 0.0;
        idx_nsv_L_.insert(i);
        idx_Dc_.erase(i);
        approx_dsol_c_[i] = 1.0;
        temp = L_coeff - sum_alpha_ratio_ * ref_dsol_[i];//second item of (6)
        approx_dsol_r_L_sq_ += temp * temp;//second item of (6)
    }

    auto end_time = sys_clk::now();
    sam_scr_time_ += static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "++++++++++++++++++++++++++++END !! sample_screening"
						<< std::endl;
	}
}

void s3ifs::feature_screening(void) 
{
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "---------------------------s3ifs::feature_screening"
						<< std::endl;
	}
    auto start_time = sys_clk::now();

    double dsol_radius =
        std::sqrt(std::max(0.0, dif_alpha_ratio_ * dif_alpha_ratio_ *
                           approx_dsol_r_sq_ - approx_dsol_r_L_sq_
                           - approx_dsol_r_R_sq_)); //formular (6)

    double Xj_Dc_sq, temp, XjTdsol_ub;
    std::vector<int> new_iaf;
    double n_beta = n_sams_ * beta_;
    for (auto &&j : idx_Fc_) 
	{
        Xj_Dc_sq = 0.0;

        for (scm_iit it(X_CM_, j); it; ++it) 	
		{
            if (idx_Dc_flag_[it.index()])
			{
                Xj_Dc_sq += it.value() * it.value();
            }
        }

        temp = (approx_dsol_c_.transpose() * X_CM_.col(j))(0);//the first item of Lemma 4
        XjTdsol_ub = std::abs(temp) + dsol_radius * std::sqrt(Xj_Dc_sq); //see Lemma 4
		
		// This is the (IFS) in Theorem 4
        if (XjTdsol_ub <= n_beta - 1e-9) {
            new_iaf.push_back(j);
        }
    }

    for (auto && j : new_iaf) 
	{
        idx_Fc_flag_[j] = 0.0;
        idx_Fc_.erase(j);
        idx_F_.insert(j);
        psol_[j] = 0.0;
        approx_psol_r_F_sq_ += ref_psol_[j] * ref_psol_[j]; //second squre of (3)
        ref_psol_[j] = 0.0;
    }

    auto end_time = sys_clk::now();
    fea_scr_time_ += static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "---------------------------END !! feature_screening"
						<< std::endl;
	}
}


void s3ifs::sifs(const bool& sample_scr_first) 
{
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>sifs"
						<< std::endl;
	}
    auto start_time = sys_clk::now();
    approx_dsol_c_ =
        (dif_alpha_ratio_ * inv_gamma_) + sum_alpha_ratio_ * ref_dsol_.array();//formular (5) 
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "oooooo@sifs@approx_dsol_c_->" << approx_dsol_c_.size() 
						<< "; " << approx_dsol_c_[0] << "	...	"
						<< approx_dsol_c_[approx_dsol_c_.size()-1]
						<< std::endl;
	}

    approx_psol_r_sq_ = ref_psol_.squaredNorm(); //first squre of (3)
    approx_dsol_r_sq_ = (ref_dsol_.array() - inv_gamma_).square().sum();//first square of (6)
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "oooooo@sifs@approx_psol_r_sq_->" << approx_psol_r_sq_
						<< std::endl;
		trainingLogFile << "oooooo@sifs@approx_dsol_r_sq_->" << approx_dsol_r_sq_
						<< std::endl;
	}
 
    //initialization
    int n_Dc = n_sams_;
    int n_Fc = n_feas_;
    int n_Dc_pre = n_Dc;
    int n_Fc_pre = n_Fc;
    int scr_iter = 0;
    
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "oooooo@sifs@n_Dc->" << n_Dc
						<< std::endl;
		trainingLogFile << "oooooo@sifs@n_Fc->" << n_Fc
						<< std::endl;
		trainingLogFile << "oooooo@sifs@n_Dc_pre->" << n_Dc_pre
						<< std::endl;
		trainingLogFile << "oooooo@sifs@n_Fc_pre->" << n_Fc_pre
						<< std::endl;
	}

    while (true)
	{
        scr_iter++;
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << "oooooo@sifs@scr_iter->" << scr_iter
							<< std::endl;
		}

        n_Dc_pre = n_Dc;
        n_Fc_pre = n_Fc;
        if (sample_scr_first) 
		{
			if (trainingLogFile && debugMode == 1)	{
				trainingLogFile << "oooooo@sifs@sample_scr_first->" << sample_scr_first
								<< std::endl;
			}
            sample_screening();
            feature_screening();
        } 
		else 
		{
			if (trainingLogFile && debugMode == 1)	{
				trainingLogFile << "oooooo@sifs@sample_scr_first->" << sample_scr_first
								<< std::endl;
			}
            feature_screening();
            sample_screening();
        }
        n_Dc = idx_Dc_.size();
        n_Fc = idx_Fc_.size();
		if (trainingLogFile && debugMode == 1)	{
			trainingLogFile << "......@sifs@n_Dc->" << n_Dc
							<< std::endl;
			trainingLogFile << "......@sifs@n_Fc->" << n_Fc
							<< std::endl;
		}

        if ((n_Dc == n_Dc_pre) && (n_Fc == n_Fc_pre)) 
		{
			if (trainingLogFile && debugMode == 1)	{
				trainingLogFile << "!!!!!!@sifs@(n_Dc == n_Dc_pre) && (n_Fc == n_Fc_pre): "
								<< n_Dc << "==" << n_Dc_pre << " && " 
								<< n_Fc << "==" << n_Fc_pre
								<< std::endl; 
			}
            break;
        }
        if (scr_max_iter_ > 0 && scr_iter >= scr_max_iter_) 
		{
			if (trainingLogFile && debugMode == 1)	{
				trainingLogFile << "!!!!!!@sifs@scr_max_iter_ > 0 && scr_iter >= scr_max_iter_: "
								<< scr_iter << "==" << scr_max_iter_
								<< std::endl; 
			}
            break;
        }
    }

    idx_Dc_vec_.clear();
    idx_Fc_vec_.clear();

    for (auto &&j : idx_Dc_) 
	{
        idx_Dc_vec_.push_back(j);
    }
    for (auto &&j : idx_Fc_) 
	{
        idx_Fc_vec_.push_back(j);
    }
    auto end_time = sys_clk::now();
    scr_time_ = static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>sifs !! END"
						<< std::endl;
	}
}

void s3ifs::ifs(void) 
{
    auto start_time = sys_clk::now();
    approx_dsol_c_ =
        (dif_alpha_ratio_ * inv_gamma_) + sum_alpha_ratio_ * ref_dsol_.array();
    approx_psol_r_sq_ = ref_psol_.squaredNorm();
    approx_dsol_r_sq_ = (ref_dsol_.array() - inv_gamma_).square().sum();
 
    feature_screening();

    idx_Fc_vec_.clear();
    for (auto &&j : idx_Fc_) {
        idx_Fc_vec_.push_back(j);
    }
    auto end_time = sys_clk::now();
    scr_time_ = static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void s3ifs::iss(void) 
{
    auto start_time = sys_clk::now();
    approx_dsol_c_ = (dif_alpha_ratio_ * inv_gamma_) + sum_alpha_ratio_ * ref_dsol_.array();
    approx_psol_r_sq_ = ref_psol_.squaredNorm();
    approx_dsol_r_sq_ = (ref_dsol_.array() - inv_gamma_).square().sum();
 
    sample_screening();

    idx_Dc_vec_.clear();
    for (auto &&j : idx_Dc_) {
        idx_Dc_vec_.push_back(j);
    }
    auto end_time = sys_clk::now();
    scr_time_ = static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void s3ifs::parse_command_line() 
{
    // default parameters
	if (trainingLogFile && debugMode == 1)	{
		trainingLogFile << "@@@@@@@@s3ifs::parse_command_line()" << std::endl;
	}
    rbu = 1.0;
    rbl = 0.05;
    nbs = 10;
    rau = 1.0;
    ral = 0.01;
    nas = 100;
    max_iter_ = 10000;
    gamma_ = 0.05;
    tol_ = 1e-9;
    chk_fre_ = 10;
    scr_max_iter_ = 0;
    alpha_ = 1e-3;
    beta_ = 1e-3;
    task = 1;
}

