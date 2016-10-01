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

#ifndef LARANK_H
#define LARANK_H

#include "Rect.h"
#include "Sample.h"

#include <vector>
#include <Eigen/Core>

#include <opencv/cv.h>

class Config;
class Features;
class Kernel;
extern std::ofstream trainingLogFile; 

#include <chrono>
using sys_clk = std::chrono::system_clock;

#include "s3ifs.h"
using SpMatRd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using scm_iit = Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator;
using srm_iit = Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;

class LaRank
{
public:
	LaRank(const Config& conf, const Features& features, const Kernel& kernel);
	~LaRank();
	
	virtual void Eval(const MultiSample& x, std::vector<double>& results);
	virtual void Update(const MultiSample& x, int y);
	
	virtual void Debug();
    SpMatRd X_;

private:

	int debugMode = 1;
	int printMode = 1;

	struct SupportPattern
	{
		std::vector<Eigen::VectorXd> x;
		std::vector<FloatRect> yv;
		std::vector<cv::Mat> images;
		int y;
		int refCount;
	};

	struct SupportPattern_tmp
	{
		std::vector<Eigen::VectorXd> x;
		std::vector<FloatRect> yv;
		std::vector<cv::Mat> images;
		int y;
		int refCount;
	};

	struct SupportVector
	{
		SupportPattern* x;
		int y;
		double b;
		double g;
		cv::Mat image;
	};

	struct SupportVector_tmp
	{
		SupportPattern_tmp* x;
		int y;
		double b;
		double g;
		double l;
		cv::Mat image;
	};
	
	const Config& m_config;
	const Features& m_features;
	const Kernel& m_kernel;
	
	std::vector<SupportPattern*> m_sps;
	std::vector<SupportPattern_tmp*> m_sps_tmp;
	std::vector<SupportVector*> m_svs;
	std::vector<SupportVector_tmp*> m_svs_tmp;

	cv::Mat m_debugImage;
	
	double m_C;
	Eigen::MatrixXd m_K;
	Eigen::MatrixXd m_K_tmp;

	inline double Loss(const FloatRect& y1, const FloatRect& y2) const
	{
		// overlap loss
		return 1.0-y1.Overlap(y2);
		// squared distance loss
		//double dx = y1.XMin()-y2.XMin();
		//double dy = y1.YMin()-y2.YMin();
		//return dx*dx+dy*dy;
	}
	
	double ComputeDual() const;

	void SMOStep(int ipos, int ineg);
	std::pair<int, double> MinGradient(int ind);
	void ProcessNew(int ind);
	void Reprocess();
	void ProcessOld();
	void Optimize();

	int AddSupportVector(SupportPattern* x, int y, double g);
	void AddSupportVector_tmp(SupportPattern_tmp* x, int y, double g, double l);
	void RemoveSupportVector(int ind);
	void RemoveSupportVectors(int ind1, int ind2);
	void SwapSupportVectors(int ind1, int ind2);
	
	void BudgetMaintenance();
	void BudgetMaintenanceRemove();

	double Evaluate(const Eigen::VectorXd& x, const FloatRect& y) const;
	double Evaluate_tmp(const Eigen::VectorXd& x, const FloatRect& y) const;
	void UpdateDebugImage();
};

#endif
