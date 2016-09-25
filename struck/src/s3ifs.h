#ifndef S3IFS_H
#define S3IFS_H

#include <sdm/lib/utils.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sdm/lib/io.hpp>
#include <random>
#include <numeric>
#include <vector>
#include <iterator>
#include <chrono>
#include <unordered_set>


extern std::ofstream trainingLogFile;

using SpMatRd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpMatCd = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using scm_iit = Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator;
using srm_iit = Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;

using sec = std::chrono::seconds;
using mil_sec = std::chrono::milliseconds;
using sys_clk = std::chrono::system_clock;

using u_set = std::unordered_set<int>;

class s3ifs
{
public:
	s3ifs(SpMatRd X_, SpMatRd D_);
	~s3ifs();

    int get_n_sams(void) const;
    int get_n_feas(void) const;

    double get_primal_obj(void) const;
    double get_dual_obj(void) const; 
    double get_duality_gap(void) const; 

    Eigen::VectorXd get_dual_sol(void) const;
    Eigen::VectorXd get_primal_sol(void) const;
    
    void set_alpha(const double& alpha, const bool& ws = true);
    void set_beta(const double& beta);
    void set_stop_tol(const double& tol);
    
    void compute_primal_obj(const bool& flag_comp_loss); // compute primal objective
    void compute_dual_obj(const bool& flag_comp_XTdsol); // compute dual objective
    void compute_duality_gap(const bool& flag_comp_loss,
                             const bool& flag_comp_XTdsol); // compute duality gap
 
    void update_psol(const bool& flag_comp_XTdsol);

    void train(void);
    void train_sifs(const int& scr_option = 0);

    void sample_screening(void);
    void feature_screening(void);
 
    void sifs(const bool& sample_scr_first = true);
    void ifs(void);
    void iss(void);

    void clear_idx(void);

    int get_n_L(void) const;
    int get_n_R(void) const;
    int get_n_F(void) const;
    int get_iter(void) const;

    double get_alpha_max(void) const;
    double get_beta_max(void) const;

    SpMatRd X_;  // \bar{X}, each row contains one sample
    SpMatRd D_;  // \bar{X}, each row contains one sample
    SpMatCd X_CM_;  // column major representation of \bar{X}
    //Eigen::ArrayXd y_;  // training labels
    
    double fea_scr_time_;
    double sam_scr_time_;
    double scr_time_;

private:
	int log_yn = 1;
	
	double rbu, rbl, rau, ral;
	int nbs, nas, max_iter, chk_fre, scr_max_iter;
	double gam, tol;
	double alpha, beta;
	int task;

	void parse_command_line();

    int n_sams_;
    int n_feas_;

    double alpha_;
    double beta_;
    double gamma_;
    double tol_;
    int max_iter_;
    int chk_fre_;
    int iter_;
 
    double alpha_max_;
    double beta_max_;
    int scr_max_iter_;

    double pobj_;
    double dobj_;
    double duality_gap_;
    double loss_;

    double inv_n_sams_;
    double inv_alpha_;
    double inv_gamma_;

    Eigen::ArrayXd one_over_XTones_;  // \bar{X}^T * ones / n
    Eigen::VectorXd psol_;
    Eigen::VectorXd dsol_;
    Eigen::VectorXd XTdsol_; // \bar{X}^T * theta / n
    Eigen::ArrayXd Xw_comp_; // 1 - \bar{X} * w
    
    Eigen::ArrayXd Xi_norm_;
    Eigen::ArrayXd Xi_norm_sq_;
    Eigen::ArrayXd Xj_norm_;
    Eigen::ArrayXd Xj_norm_sq_;
    
    std::vector<int> all_ins_index_;
    std::vector<int> all_fea_index_;
    u_set idx_nsv_L_;
    u_set idx_nsv_R_;
    u_set idx_Dc_;
    u_set idx_F_;
    u_set idx_Fc_;
    std::vector<int> idx_Dc_vec_;
    std::vector<int> idx_Fc_vec_;
    Eigen::VectorXd idx_Dc_flag_;
    Eigen::VectorXd idx_Fc_flag_;
    
    double ref_alpha_; // alpha_0
    double dif_alpha_ratio_; // (alpha - alpha_0) / alpha
    double sum_alpha_ratio_; // (alpha + alpha_0) / alpha
    Eigen::VectorXd ref_psol_;  // reference primal solution w_0,
                                //also center of primal optimum estimation
    Eigen::VectorXd ref_dsol_;
    Eigen::VectorXd approx_dsol_c_;  // center of dual optimum estimation
    double approx_dsol_r_sq_;  // square of radius of dual optimum estimation
    double approx_dsol_r_L_sq_;
    double approx_dsol_r_R_sq_;
    double approx_psol_r_sq_;  // |w_0|^2
    double approx_psol_r_F_sq_;  // |w_0(F)|^2

};

template <typename _Tp> inline _Tp val_sign(_Tp val) {
    return 1.0 - (val <= 0.0) - (val < 0.0);
}

#endif
