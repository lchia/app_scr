/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "LaRank.h"

#include "Config.h"
#include "Features.h"
#include "Kernels.h"
#include "Sample.h"
#include "Rect.h"
#include "GraphUtils/GraphUtils.h"

#include <Eigen/Core>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv/highgui.h>

static const int kTileSize = 30;
using namespace cv;

using namespace std;
using namespace Eigen;

static const int kMaxSVs = 2000; // TODO (only used when no budget)


LaRank::LaRank(const Config& conf, const Features& features, const Kernel& kernel) :
	m_config(conf),
	m_features(features),
	m_kernel(kernel),
	m_C(conf.svmC)
{
	int N = conf.svmBudgetSize > 0 ? conf.svmBudgetSize+2 : kMaxSVs;
	m_K = MatrixXd::Zero(N, N);
	m_K_tmp = MatrixXd::Zero(N, N);
	m_debugImage = Mat(800, 600, CV_8UC3);
	m_debugImage_tmp = Mat(800, 600, CV_8UC3);
}

LaRank::~LaRank()
{
}

double LaRank::Evaluate(const Eigen::VectorXd& x, const FloatRect& y) const
{
	double f = 0.0;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		const SupportVector& sv = *m_svs[i];
		f += sv.b*m_kernel.Eval(x, sv.x->x[sv.y]);
	}
	return f;
}

double LaRank::Evaluate_tmp(const Eigen::VectorXd& x, const FloatRect& y) const
{
	double f = 0.0;
	for (int i = 0; i < (int)m_svs_tmp.size(); ++i)
	{
		const SupportVector_tmp& sv_tmp = *m_svs_tmp[i]; 
		f += sv_tmp.b*m_kernel.Eval(x, sv_tmp.x->x[sv_tmp.y]);
	}
	return f;
}

double LaRank::Evaluate_tmp_score(const Eigen::VectorXd& x, Eigen::VectorXd psol_) const
{
	double f = 0.0;
	//cout << "	....................................Score_TMP" << endl;
	//cout << "	psol_.size()=" << psol_.size() << endl;
	//cout << "	x.size()=" << x.size() << endl;
	//cout << "	score: f=";
	
	for(int i=0; i<x.size(); i++) 
	{	
		//cout << x[i] << "*" << psol_[i] << "+";
		f += x[i] * psol_[i];
	}
	
	//cout << "= " << f << endl;

	return f;
}
void LaRank::Eval(const MultiSample& sample, std::vector<double>& results,  
					  Eigen::VectorXd psol_, std::vector<double>& results_tmp)
{
	const FloatRect& centre(sample.GetRects()[0]);
	vector<VectorXd> fvs;
	const_cast<Features&>(m_features).Eval(sample, fvs);
	results.resize(fvs.size());
	results_tmp.resize(fvs.size());
	for (int i = 0; i < (int)fvs.size(); ++i)
	{
		// express y in coord frame of centre sample
		FloatRect y(sample.GetRects()[i]);
		y.Translate(-centre.XMin(), -centre.YMin()); 
		results[i] = Evaluate(fvs[i], y);
		results_tmp[i] = Evaluate_tmp_score(fvs[i], psol_);
	}
}

void LaRank::Update(const MultiSample& sample, int y)
{
	// add new support pattern
	SupportPattern* sp = new SupportPattern;
	SupportPattern_tmp* sp_tmp = new SupportPattern_tmp;
	const vector<FloatRect>& rects = sample.GetRects();
	FloatRect centre = rects[y];
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		// express r in coord frame of centre sample
		FloatRect r = rects[i];
		r.Translate(-centre.XMin(), -centre.YMin());
		sp->yv.push_back(r);
		sp_tmp->yv.push_back(r);
		if (!m_config.quietMode && m_config.debugMode)
		{
			// store a thumbnail for each sample
			Mat im(kTileSize, kTileSize, CV_8UC1);
			IntRect rect = rects[i]; 
			cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
			cv::resize(sample.GetImage().GetImage(0)(roi), im, im.size());
			sp->images.push_back(im);
			sp_tmp->images.push_back(im);
		}
	}
	// evaluate features for each sample
	sp->x.resize(rects.size());
	sp_tmp->x.resize(rects.size());

	cout << ".....................LaRank::Update................12.." << endl;
	cout << ".....................LaRank::Update................sample.GetRects().size()=" << sample.GetRects().size() << endl;
	cout << ".....................LaRank::Update................sp->x.size()=" << sp->x.size() << endl;
	const_cast<Features&>(m_features).Eval(sample, sp->x);

	cout << ".....................LaRank::Update................13.." << endl;
	const_cast<Features&>(m_features).Eval(sample, sp_tmp->x);

	cout << ".....................LaRank::Update................14.." << endl;
	sp->y = y;
	sp->refCount = 0;
	sp_tmp->y = y;
	sp_tmp->refCount = 0;
	m_sps.push_back(sp); 
	m_sps_tmp.push_back(sp_tmp); 

	cout << ".....................LaRank::Update....Gate of Screening." << endl;
	/**************************************************************************
	 ***	Gate of Screening, Gate of Screening, Gate of Screening 	*** 	 
	/**************************************************************************/
	if (trainingLogFile && debugMode) {
		trainingLogFile << "------------------------------------------" << endl;
		trainingLogFile << "-----------------------Gate of 'screening'" << endl;
		trainingLogFile << "          	=>Generate X_" << std::endl;
		trainingLogFile << "Update: Adding " << (int)sp_tmp->yv.size()  
						<< " rects into support vector set." << std::endl;;
	}  
	FloatRect initRect = sp_tmp->yv[0];
	int psNum = 0, nsNum = 0;
	for (int i = 0; i < (int)sp_tmp->yv.size(); i++)
	{
		double grad = - Loss(sp_tmp->yv[i], sp_tmp->yv[sp_tmp->y]) 
					  - Evaluate_tmp(sp_tmp->x[i], sp_tmp->yv[i]); 

		//compute IoU
		FloatRect currRect = sp_tmp->yv[i];
		double IoU = initRect.Overlap(currRect);  
	 	int l = (IoU > 0.7) ? 1 : -1;
		if (l == 1)	
		{
			//cout << i << "-th rect; IoU = " << IoU 
			//	 << "; l = " << l << std::endl;
			psNum++;
			AddSupportVector_tmp(sp_tmp, i, grad, l);
		}
		else if (l == -1)	
		{
			//cout << i << "-th rect; IoU = " << IoU 
			//	 << "; l = " << l << std::endl;
			nsNum++;
			AddSupportVector_tmp(sp_tmp, i, grad, l);
		}
		/**
		cout << "@@<m_sps,m_sps_tmp>=" << m_sps.size()  << "," 
			 << m_sps_tmp.size()
		 	 << "; <m_svs,m_svs_tmp>=: " << m_svs.size() << "," 
			 << m_svs_tmp.size() << std::endl;
		**/
	}
	if (trainingLogFile && debugMode) { 
		trainingLogFile << "*********Sample <Positive,Negative>=<" 
						<< psNum << "," << nsNum <<">" << std::endl;
	}
	cout << ".....................LaRank::Update...Sample <Positive,Negative>=<" 
						<< psNum << "," << nsNum <<">" << std::endl;

	//Put all the new rects and existing svs together -> X_
	int featureDim = (int)sp_tmp->x[sp_tmp->y].size();
	int sampleNum = (int)m_svs_tmp.size();
	X_.resize(sampleNum, featureDim); 
	for (int i = 0; i < sampleNum; ++i)
	{
		const SupportVector_tmp* svi = m_svs_tmp[i];	//the i-th sv
		Eigen::ArrayXd svFea = svi->x->x[svi->y];	//the corresponding feature
		double l = svi->l;
		for (int j = 0; j < featureDim; ++j) {	
			X_.insert(i,j)  = svFea[j] * l;
		}
	}
	X_.makeCompressed();

	cout << ".....................LaRank::Update..s3ifs solver(X_)" << std::endl;
	s3ifs solver(X_); //Simutaneous Inactive Samples and features Screening
	double beta_ub = solver.get_beta_max() * solver.rbu;
	double beta_lb = solver.get_beta_max() * solver.rbl;
	double beta_log_inter = (log10(beta_ub) - log10(beta_lb)) /
		static_cast<double>(solver.nbs);
	double beta_now;

	cout << ".....................LaRank::Update..Solver: bata.alpha.." << endl;
	for(int i = solver.nbs - 1; i >= 0; --i)
	{
		if (trainingLogFile && debugMode) {
			trainingLogFile << "____ " << i << "-th:"
							<< "beta=" << solver.beta_ 
							<< "	alpha=" << solver.alpha_ << "	";
		}
		beta_now = std::pow(10.0, log10(beta_lb) + 0.5 * beta_log_inter
					+ i * beta_log_inter);
		solver.set_beta(beta_now);
		
		double alpha_ub = solver.get_alpha_max() * solver.rau;
		double alpha_lb = solver.get_alpha_max() * solver.ral;
		double alpha_log_inter = (log10(alpha_ub) - log10(alpha_lb)) /
			static_cast<double>(solver.nas);
		double alpha_now;
		cout << "..............................LaRank::Update.@beta..." << endl;
		for (int j = solver.nas -1; j > 0; --j)
		{
			//Set alpha
			alpha_now = std::pow(10.0, log10(alpha_lb) + (j+1)*alpha_log_inter);

			if (j == solver.nas - 1) 
				solver.set_alpha(alpha_now, false); 
			else 
				solver.set_alpha(alpha_now, true); 
			
			Eigen::VectorXd psol, dsol;
			Eigen::ArrayXd Xw_comp;
			int ias_R, ias_L, iaf;

			int pzn0 = 0; //primal optimum zero number
			int dzn0 = 0; //dual optimum zero number
			for(int kk=0; kk < solver.psol_.size(); kk++) 
				{ if (solver.psol_[kk] == 0) pzn0++; }
			for(int kk=0; kk < solver.dsol_.size(); kk++)
				{ if (solver.dsol_[kk] == 0) dzn0++; }

			switch (solver.task) {
			case 1: { 
				auto start_time = sys_clk::now();
				solver.train_sifs(1);
				auto end_time = sys_clk::now();
		
				double train_rt = 1e-3 * static_cast<double>(
					std::chrono::duration_cast<mil_sec>(end_time-start_time).count());
				break; 
				}
			} 
			if (trainingLogFile && debugMode && j%20==0) {
				int pzn1 = 0; //primal optimum zero number
				int dzn1 = 0; //dual optimum zero number
				for(int kk=0; kk < solver.psol_.size(); kk++) 
					{ if (solver.psol_[kk] == 0) pzn1++; }
				for(int kk=0; kk < solver.dsol_.size(); kk++)
					{ if (solver.dsol_[kk] == 0) dzn1++; }
				trainingLogFile << pzn1 << "/" << dzn0 << ",";
			}
		} 

		//--------------------------------------------------------------
		//-- Remove svs with dsol_[i] = 0
		if (trainingLogFile && debugMode) {
			trainingLogFile << "	@m_svs:" << m_svs_tmp.size() << "/";
			cout << "****************************@" << i << "-th beta: " 
			 	 << m_svs_tmp.size() << "/";
		}
		RemoveSupportVector_tmp(solver.dsol_);
		if (trainingLogFile && debugMode) { 
			trainingLogFile << m_svs_tmp.size() << std::endl;
			cout << m_svs_tmp.size() << std::endl;
		} 
 
		//--------------------------------------------------------------
		//-- print the optimum of the Primal && dual
		if (!trainingLogFile && debugMode) 
		{
			trainingLogFile << "	@psol_[" << solver.psol_.size() << "]:";
			for (int kk = 0; kk < solver.psol_.size(); kk++) 	
			{
				if (kk%1==0 || kk == solver.psol_.size()-1) 
				{
					trainingLogFile << solver.psol_[kk] << ",";
				}
			}
			trainingLogFile << std::endl;
		}

		if (!trainingLogFile && debugMode) 
		{
			trainingLogFile << "	@dsol_[" << solver.dsol_.size() << "]:";
			for (int kk = 0; kk < solver.dsol_.size(); kk++) 	
			{
				if (kk%1==0 || kk == solver.dsol_.size()-1) {
					trainingLogFile << solver.dsol_[kk] << ",";
				}
			}
			trainingLogFile << std::endl;
		}
	}
	trainingLogFile << std::endl; 
	//--------------------------------------------------------------
	//-- Update primal optimum: psol_
	psol_ = solver.psol_;

 	/**************************************************************************
	 ** Struck, ** Struck, ** Struck, ** Struck, ** Struck, ** Struck, ********
	 **************************************************************************/
	ProcessNew((int)m_sps.size()-1);
	BudgetMaintenance();
	
	for (int i = 0; i < 10; ++i)
	{
		Reprocess();
		BudgetMaintenance();
	}

	//==========================================================================
	//== Update Done. Compare the sps & svs
	if (trainingLogFile && debugMode) { 
		trainingLogFile << "..........................m_sps/m_sps_tmp: " 
			 << m_sps.size()  << "/" << m_sps_tmp.size()
		 	 << "; m_svs/m_svs_tmp: " << m_svs.size() << "/" << m_svs_tmp.size()
			 << std::endl;
	}
}

void LaRank::BudgetMaintenance()
{
	if (m_config.svmBudgetSize > 0)
	{
		while ((int)m_svs.size() > m_config.svmBudgetSize)
		{
			BudgetMaintenanceRemove();
		}
	}
}

void LaRank::Reprocess()
{
	ProcessOld();
	for (int i = 0; i < 10; ++i)
	{
		Optimize();
	}
}

double LaRank::ComputeDual() const
{
	double d = 0.0;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		const SupportVector* sv = m_svs[i];
		d -= sv->b*Loss(sv->x->yv[sv->y], sv->x->yv[sv->x->y]);
		for (int j = 0; j < (int)m_svs.size(); ++j)
		{
			d -= 0.5*sv->b*m_svs[j]->b*m_K(i,j);
		}
	}
	return d;
}

void LaRank::SMOStep(int ipos, int ineg)
{
	if (ipos == ineg) return;

	SupportVector* svp = m_svs[ipos];
	SupportVector* svn = m_svs[ineg];
	assert(svp->x == svn->x);
	SupportPattern* sp = svp->x;

#if VERBOSE
	cout << "SMO: gpos:" << svp->g << " gneg:" << svn->g << endl;
#endif	
	if ((svp->g - svn->g) < 1e-5)
	{
#if VERBOSE
		cout << "SMO: skipping" << endl;
#endif		
	}
	else
	{
		double kii = m_K(ipos, ipos) + m_K(ineg, ineg) - 2*m_K(ipos, ineg);
		double lu = (svp->g-svn->g)/kii;
		// no need to clamp against 0 since we'd have skipped in that case
		double l = min(lu, m_C*(int)(svp->y == sp->y) - svp->b);

		svp->b += l;
		svn->b -= l;

		// update gradients
		for (int i = 0; i < (int)m_svs.size(); ++i)
		{
			SupportVector* svi = m_svs[i];
			svi->g -= l*(m_K(i, ipos) - m_K(i, ineg));
		}
#if VERBOSE
		cout << "SMO: " << ipos << "," << ineg << " -- " << svp->b << "," << svn->b << " (" << l << ")" << endl;
#endif		
	}
	
	// check if we should remove either sv now
	
	if (fabs(svp->b) < 1e-8)
	{
		RemoveSupportVector(ipos);
		if (ineg == (int)m_svs.size())
		{
			// ineg and ipos will have been swapped during sv removal
			ineg = ipos;
		}
	}

	if (fabs(svn->b) < 1e-8)
	{
		RemoveSupportVector(ineg);
	}
}

pair<int, double> LaRank::MinGradient(int ind)
{
	const SupportPattern* sp = m_sps[ind];
	pair<int, double> minGrad(-1, DBL_MAX);
	for (int i = 0; i < (int)sp->yv.size(); ++i)
	{
		double grad = -Loss(sp->yv[i], sp->yv[sp->y]) - Evaluate(sp->x[i], sp->yv[i]);
		if (grad < minGrad.second)
		{
			minGrad.first = i;
			minGrad.second = grad;
		}
	}
	return minGrad;
}

void LaRank::ProcessNew(int ind)
{
	// gradient is -f(x,y) since loss=0
	int ip = AddSupportVector(m_sps[ind], m_sps[ind]->y, -Evaluate(m_sps[ind]->x[m_sps[ind]->y],m_sps[ind]->yv[m_sps[ind]->y]));

	pair<int, double> minGrad = MinGradient(ind);
	int in = AddSupportVector(m_sps[ind], minGrad.first, minGrad.second);

	SMOStep(ip, in);
}

void LaRank::ProcessOld()
{
	if (m_sps.size() == 0) return;

	// choose pattern to process
	int ind = rand() % m_sps.size();

	// find existing sv with largest grad and nonzero beta
	int ip = -1;
	double maxGrad = -DBL_MAX;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		const SupportVector* svi = m_svs[i];
		if (svi->g > maxGrad && svi->b < m_C*(int)(svi->y == m_sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
	}
	assert(ip != -1);
	if (ip == -1) return;

	// find potentially new sv with smallest grad
	pair<int, double> minGrad = MinGradient(ind);
	int in = -1;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		if (m_svs[i]->y == minGrad.first)
		{
			in = i;
			break;
		}
	}
	if (in == -1)
	{
		// add new sv
		in = AddSupportVector(m_sps[ind], minGrad.first, minGrad.second);
	}

	SMOStep(ip, in);
}

void LaRank::Optimize()
{ 
	if (m_sps.size() == 0) return;
	
	// choose pattern to optimize
	int ind = rand() % m_sps.size();
	//cout << "ind=" << ind << "; m_sps.size()=" << m_sps.size() << std::endl;

	int ip = -1;
	int in = -1;
	double maxGrad = -DBL_MAX;
	double minGrad = DBL_MAX;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		const SupportVector* svi = m_svs[i];
		if (svi->g > maxGrad && svi->b < m_C*(int)(svi->y == m_sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
		if (svi->g < minGrad)
		{
			in = i;
			minGrad = svi->g;
		} 
		/**
		cout << "	<svi->b, **; m_C, svi->y, m_sps[ind]->y>= " 
			 << svi->b << "," << m_C*(int)(svi->y == m_sps[ind]->y)
			 << ";" << m_C << "," << svi->y << "," << m_sps[ind]->y << std::endl;

		cout << "	<minGrad, svi->g, maxGrad>=<" << minGrad << ","
			 << svi->g << "," << maxGrad << std::endl;
		**/
	}
	//cout << "	<ip, in>=<" << ip << "," << in << ">" << std::endl;

	assert(ip != -1 && in != -1);
	if (ip == -1 || in == -1)
	{
		// this shouldn't happen
		cout << "	^_^@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		return;
	}

	SMOStep(ip, in);
}

int LaRank::AddSupportVector(SupportPattern* x, int y, double g)
{
	SupportVector* sv = new SupportVector;
	sv->b = 0.0;
	sv->x = x;
	sv->y = y;
	sv->g = g;

	int ind = (int)m_svs.size();
	m_svs.push_back(sv);
	x->refCount++; 

#if VERBOSE
	cout << "Adding SV: " << ind << endl;
#endif

	// update kernel matrix
	for (int i = 0; i < ind; ++i)
	{
		m_K(i,ind) = m_kernel.Eval(m_svs[i]->x->x[m_svs[i]->y], x->x[y]);
		m_K(ind,i) = m_K(i,ind);
	} 
	m_K(ind,ind) = m_kernel.Eval(x->x[y]);

	return ind;
}


void LaRank::AddSupportVector_tmp(SupportPattern_tmp* x, int y, double g, double l)
{
	SupportVector_tmp* sv_tmp = new SupportVector_tmp;
	sv_tmp->b = 0.0;
	sv_tmp->x = x;
	sv_tmp->y = y;
	sv_tmp->g = g;
	sv_tmp->l = l;

	int ind = (int)m_svs_tmp.size();
	m_svs_tmp.push_back(sv_tmp);
	x->refCount++;

#if VERBOSE
	cout << "Adding SV_tmp: " << ind << endl;
#endif
	// update kernel matrix
	for (int i = 0; i < ind; ++i)
	{
		//cout << "m_K_tmp[" << i << "," << ind << "]="
		//	 << m_kernel.Eval(m_svs_tmp[i]->x->x[m_svs_tmp[i]->y], x->x[y])
		//	 << std::endl;
		m_K_tmp(i,ind) = m_kernel.Eval(m_svs_tmp[i]->x->x[m_svs_tmp[i]->y], x->x[y]);
		m_K_tmp(ind,i) = m_K_tmp(i,ind);
	}

	//cout << "m_K_tmp[" << ind << "," << ind << "]="
	//	 << m_kernel.Eval(x->x[y])
	//	 << std::endl;
	m_K_tmp(ind,ind) = m_kernel.Eval(x->x[y]);
}

void LaRank::SwapSupportVectors(int ind1, int ind2)
{
	SupportVector* tmp = m_svs[ind1];
	m_svs[ind1] = m_svs[ind2];
	m_svs[ind2] = tmp;
	
	VectorXd row1 = m_K.row(ind1);
	m_K.row(ind1) = m_K.row(ind2);
	m_K.row(ind2) = row1;
	
	VectorXd col1 = m_K.col(ind1);
	m_K.col(ind1) = m_K.col(ind2);
	m_K.col(ind2) = col1;
}

void LaRank::SwapSupportVectors_tmp(int ind1, int ind2)
{
	SupportVector_tmp* tmp = m_svs_tmp[ind1];
	m_svs_tmp[ind1] = m_svs_tmp[ind2];
	m_svs_tmp[ind2] = tmp;
	
	VectorXd row1 = m_K_tmp.row(ind1);
	m_K_tmp.row(ind1) = m_K_tmp.row(ind2);
	m_K_tmp.row(ind2) = row1;
	
	VectorXd col1 = m_K_tmp.col(ind1);
	m_K_tmp.col(ind1) = m_K_tmp.col(ind2);
	m_K_tmp.col(ind2) = col1;
}

void LaRank::RemoveSupportVector(int ind)
{
#if VERBOSE
	cout << "Removing SV: " << ind << endl;
#endif	

	m_svs[ind]->x->refCount--;
	if (m_svs[ind]->x->refCount == 0)
	{
		// also remove the support pattern
		for (int i = 0; i < (int)m_sps.size(); ++i)
		{
			if (m_sps[i] == m_svs[ind]->x)
			{
				delete m_sps[i];
				m_sps.erase(m_sps.begin()+i);
				break;
			}
		}
	}

	// make sure the support vector is at the back, this
	// lets us keep the kernel matrix cached and valid
	if (ind < (int)m_svs.size()-1)
	{
		SwapSupportVectors(ind, (int)m_svs.size()-1);
		ind = (int)m_svs.size()-1;
	}
	delete m_svs[ind];
	m_svs.pop_back();
}

void LaRank::RemoveSupportVector_tmp(Eigen::VectorXd dsol_)
{
#if VERBOSE
	cout << "Removing i-th SV with dsol_[i]=0 " << endl;
#endif 
	//Update the beta of each sv
	assert(m_svs_tmp.size() == dsol_.size());
	for (int kk=0; kk<m_svs_tmp.size()-1; kk++)
	{
		m_svs_tmp[kk]->b = dsol_[kk];
		//cout << m_svs_tmp[kk]->b << ",";
	} 
	//cout << endl;

	//Delete SVs with sv->b is 0
	int idx = 0;
	while(idx < m_svs_tmp.size())
	{
		//cout << idx << "/" << m_svs_tmp.size() << ", ";
		if (m_svs_tmp[idx]->b > 1e-6)
		{
			idx++;
		}
		else
		{
			m_svs_tmp[idx]->x->refCount--;
			if (m_svs_tmp[idx]->x->refCount == 0)
			{
				// also remove the support pattern
				for (int i = 0; i < (int)m_sps_tmp.size(); ++i)
				{
					if (m_sps_tmp[i] == m_svs_tmp[idx]->x)
					{
						delete m_sps_tmp[i];
						m_sps_tmp.erase(m_sps_tmp.begin()+i);
						break;
					}
				}
			}
			// make sure the support vector is at the back, this
			// lets us keep the kernel matrix cached and valid 
			if (idx < (int)m_svs_tmp.size()-1)
			{
				SwapSupportVectors_tmp(idx, (int)m_svs_tmp.size()-1);
			}
			delete m_svs_tmp[(int)m_svs_tmp.size()-1];
			m_svs_tmp.pop_back();
		}
	}
	//cout << endl << endl;
}

void LaRank::BudgetMaintenanceRemove()
{
	// find negative sv with smallest effect on discriminant function if removed
	double minVal = DBL_MAX;
	int in = -1;
	int ip = -1;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->b < 0.0)
		{
			// find corresponding positive sv
			int j = -1;
			for (int k = 0; k < (int)m_svs.size(); ++k)
			{
				if (m_svs[k]->b > 0.0 && m_svs[k]->x == m_svs[i]->x)
				{
					j = k;
					break;
				}
			}
			double val = m_svs[i]->b*m_svs[i]->b*(m_K(i,i) + m_K(j,j) - 2.0*m_K(i,j));
			if (val < minVal)
			{
				minVal = val;
				in = i;
				ip = j;
			}
		}
	}

	// adjust weight of positive sv to compensate for removal of negative
	m_svs[ip]->b += m_svs[in]->b;

	// remove negative sv
	RemoveSupportVector(in);
	if (ip == (int)m_svs.size())
	{
		// ip and in will have been swapped during support vector removal
		ip = in;
	}
	
	if (m_svs[ip]->b < 1e-8)
	{
		// also remove positive sv
		RemoveSupportVector(ip);
	}

	// update gradients
	// TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		SupportVector& svi = *m_svs[i];
		svi.g = -Loss(svi.x->yv[svi.y],svi.x->yv[svi.x->y]) - Evaluate(svi.x->x[svi.y], svi.x->yv[svi.y]);
	}	
}

void LaRank::Debug()
{
	cout << m_sps.size() << "/" << m_svs.size() << " Struck: support patterns/vectors" << endl;
	cout << m_sps_tmp.size() << "/" << m_svs_tmp.size() << " Screening: support patterns/vectors" << endl;
	UpdateDebugImage();
	//cout << "..............................UpdateDebugImage()" << endl;
	UpdateDebugImage_tmp();
	//cout << "..............................UpdateDebugImage_tmp()" << endl;
	imshow("learner", m_debugImage);
	imshow("learner_screening", m_debugImage_tmp);
}

void LaRank::UpdateDebugImage()
{
	m_debugImage.setTo(0);
	
	int n = (int)m_svs.size();
	
	if (n == 0) return;
	
	const int kCanvasSize = 600;
	int gridSize = (int)sqrtf((float)(n-1)) + 1;
	int tileSize = (int)((float)kCanvasSize/gridSize);
	
	if (tileSize < 5)
	{
		cout << "too many support vectors to display" << endl;
		return;
	}
	
	Mat temp(tileSize, tileSize, CV_8UC1);
	int x = 0;
	int y = 0;
	int ind = 0;
	float vals[kMaxSVs];
	memset(vals, 0, sizeof(float)*n);
	int drawOrder[kMaxSVs];
	
	for (int set = 0; set < 2; ++set)
	{
		for (int i = 0; i < n; ++i)
		{
			if (((set == 0) ? 1 : -1)*m_svs[i]->b < 0.0) continue;
			
			drawOrder[ind] = i;
			vals[ind] = (float)m_svs[i]->b;
			++ind;
			
			Mat I = m_debugImage(cv::Rect(x, y, tileSize, tileSize));
			resize(m_svs[i]->x->images[m_svs[i]->y], temp, temp.size());
			cvtColor(temp, I, CV_GRAY2RGB);
			double w = 1.0;
			rectangle(I, Point(0, 0), Point(tileSize-1, tileSize-1), (m_svs[i]->b > 0.0) ? CV_RGB(0, (uchar)(255*w), 0) : CV_RGB((uchar)(255*w), 0, 0), 3);
			x += tileSize;
			if ((x+tileSize) > kCanvasSize)
			{
				y += tileSize;
				x = 0;
			}
		}
	}
	
	const int kKernelPixelSize = 2;
	int kernelSize = kKernelPixelSize*n;
	
	double kmin = m_K.minCoeff();
	double kmax = m_K.maxCoeff();
	
	if (kernelSize < m_debugImage.cols && kernelSize < m_debugImage.rows)
	{
		Mat K = m_debugImage(cv::Rect(m_debugImage.cols-kernelSize, m_debugImage.rows-kernelSize, kernelSize, kernelSize));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				Mat Kij = K(cv::Rect(j*kKernelPixelSize, i*kKernelPixelSize, kKernelPixelSize, kKernelPixelSize));
				uchar v = (uchar)(255*(m_K(drawOrder[i], drawOrder[j])-kmin)/(kmax-kmin));
				Kij.setTo(Scalar(v, v, v));
			}
		}
	}
	else
	{
		kernelSize = 0;
	}
	
	
	Mat I = m_debugImage(cv::Rect(0, m_debugImage.rows - 200, m_debugImage.cols-kernelSize, 200));
	I.setTo(Scalar(255,255,255));
	IplImage II = I;
	setGraphColor(0);
	drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
}

void LaRank::UpdateDebugImage_tmp()
{
	//cout << "......................................UpdateDebugImage_tmp: "<< 0 << endl;
	m_debugImage_tmp.setTo(0);
	
	//cout << "......................................UpdateDebugImage_tmp: "<< 1 << endl;
	int n = (int)m_svs_tmp.size(); 
	//cout << "......................................UpdateDebugImage_tmp: "<< n << endl;
	if (n == 0) return;
	
	const int kCanvasSize = 600;
	int gridSize = (int)sqrtf((float)(n-1)) + 1;
	int tileSize = (int)((float)kCanvasSize/gridSize); 
	//cout << "......................................gridSize: "<< gridSize << endl;
	//cout << "......................................tileSize: "<< tileSize << endl;

	if (tileSize < 5)
	{
		cout << "too many support vectors to display" << endl;
		return;
	}
	
	Mat temp(tileSize, tileSize, CV_8UC1);
	//cout << "......................................UpdateDebugImage_tmp: "<< 2 << endl;
	int x = 0;
	int y = 0;
	int ind = 0;
	float vals[kMaxSVs];
	memset(vals, 0, sizeof(float)*n);
	int drawOrder[kMaxSVs]; 
	//cout << "......................................UpdateDebugImage_tmp: "<< 3 << endl;

	for (int set = 0; set < 2; ++set)
	{
		for (int i = 0; i < n; ++i)
		{
			if (((set == 0) ? 1 : -1)*m_svs_tmp[i]->b < 0.0) continue;//??????????????
			
			drawOrder[ind] = i;
			vals[ind] = (float)m_svs_tmp[i]->b;
			++ind;
			
			//cout << "......................................UpdateDebugImage_tmp: "<< 3.1 << endl;
			//cout << "......................................[x,y,tileSize,tileSize]: "
			//	 << x << "," << y << "," << tileSize << ","  << endl;
			Mat I = m_debugImage_tmp(cv::Rect(x, y, tileSize, tileSize));
			//cout << "......................................UpdateDebugImage_tmp: "<< 3.2 << endl;
			resize(m_svs_tmp[i]->x->images[m_svs_tmp[i]->y], temp, temp.size());
			//cout << "......................................UpdateDebugImage_tmp: "<< 3.3 << endl;
			cvtColor(temp, I, CV_GRAY2RGB);
			//cout << "......................................UpdateDebugImage_tmp: "<< 3.4 << endl;
			double w = 1.0;
			//cout << "......................................UpdateDebugImage_tmp: "<< 4 << endl;
			rectangle(I, Point(0, 0), Point(tileSize-1, tileSize-1), (m_svs_tmp[i]->b > 0.0) ? CV_RGB(0, (uchar)(255*w), 0) : CV_RGB((uchar)(255*w), 0, 0), 3);
			x += tileSize;
			if ((x+tileSize) > kCanvasSize)
			{
				y += tileSize;
				x = 0;
			}
		}
	}
	//cout << "......................................UpdateDebugImage_tmp: "<< 5 << endl;
	const int kKernelPixelSize = 2;
	int kernelSize = kKernelPixelSize*n;
	
	double kmin = m_K.minCoeff();
	double kmax = m_K.maxCoeff();
	
	if (kernelSize < m_debugImage_tmp.cols && kernelSize < m_debugImage_tmp.rows)
	{
		//cout << "......................................UpdateDebugImage_tmp: "<< 6 << endl;
		Mat K = m_debugImage_tmp(cv::Rect(m_debugImage_tmp.cols-kernelSize, m_debugImage_tmp.rows-kernelSize, kernelSize, kernelSize));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				Mat Kij = K(cv::Rect(j*kKernelPixelSize, i*kKernelPixelSize, kKernelPixelSize, kKernelPixelSize));
				uchar v = (uchar)(255*(m_K(drawOrder[i], drawOrder[j])-kmin)/(kmax-kmin));
				Kij.setTo(Scalar(v, v, v));
			}
		}
	}
	else
	{
		kernelSize = 0;
	}
	
	//cout << "......................................UpdateDebugImage_tmp: "<< 7 << endl;
	Mat I = m_debugImage_tmp(cv::Rect(0, m_debugImage_tmp.rows - 200, m_debugImage_tmp.cols-kernelSize, 200));
	I.setTo(Scalar(255,255,255));
	//cout << "......................................UpdateDebugImage_tmp: "<< 8 << endl;
	IplImage II = I;
	setGraphColor(0);
	drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
}
