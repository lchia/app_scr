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

#include "Tracker.h"
#include "Config.h"
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"
#include "GraphUtils/GraphUtils.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"

#include "Kernels.h"

#include "LaRank.h"

#include <opencv/cv.h>
//#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>

#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;
 
Tracker::Tracker(const Config& conf) :
	m_config(conf),
	m_initialised(false),
	m_pLearner(0),
	m_debugImage(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	m_debugImage_tmp(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	m_needsIntegralImage(false)
{
	Reset();
}

Tracker::~Tracker()
{
	delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
}

void Tracker::Reset()
{
	m_initialised = false;
	m_debugImage.setTo(0);
	m_debugImage_tmp.setTo(0);
	if (m_pLearner) delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();
	
	m_needsIntegralImage = false;
	m_needsIntegralHist = false;
	
	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
	for (int i = 0; i < numFeatures; ++i)
	{
		switch (m_config.features[i].feature)
		{
		case Config::kFeatureTypeHaar:
			m_features.push_back(new HaarFeatures(m_config));
			m_needsIntegralImage = true;
			break;			
		case Config::kFeatureTypeRaw:
			m_features.push_back(new RawFeatures(m_config));
			break;
		case Config::kFeatureTypeHistogram:
			m_features.push_back(new HistogramFeatures(m_config));
			m_needsIntegralHist = true;
			break;
		}
		featureCounts.push_back(m_features.back()->GetCount());
		
		switch (m_config.features[i].kernel)
		{
		case Config::kKernelTypeLinear:
			m_kernels.push_back(new LinearKernel());
			break;
		case Config::kKernelTypeGaussian:
			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
			break;
		case Config::kKernelTypeIntersection:
			m_kernels.push_back(new IntersectionKernel());
			break;
		case Config::kKernelTypeChi2:
			m_kernels.push_back(new Chi2Kernel());
			break;
		}
	}
	
	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);
		
		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);		
	}
	
	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());
}
	

void Tracker::Initialise(const cv::Mat& frame, FloatRect bb)
{
	m_bb = IntRect(bb);
	m_bb_tmp = IntRect(bb);
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)
	{
		UpdateLearner(image);
	}
	m_initialised = true;
}

void Tracker::Track(const cv::Mat& frame)
{
	assert(m_initialised);
	
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	
	vector<FloatRect> rects = Sampler::PixelSamples(m_bb, m_config.searchRadius);
	
	vector<FloatRect> keptRects;
	keptRects.reserve(rects.size());
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}
	
	MultiSample sample(image, keptRects);
	
	cout << "		....compute scores" << endl;
	vector<double> scores;
	vector<double> scores_tmp; 
	m_pLearner->Eval(sample, scores, m_pLearner->psol_, scores_tmp); 

			 
	cout << "	rects Number: " << (int)keptRects.size() << endl;
	int idx = 0;
	for (int k=0; k<scores_tmp.size(); k++)
	{	
		if (scores_tmp[k] > 0) idx++;	
	}
	cout << "	Positive sample Number: " << idx << endl;


	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i; 
		}
	}
	double bestScore_tmp = -DBL_MAX;
	int bestInd_tmp = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores_tmp[i] > bestScore)
		{
			bestScore_tmp = scores[i];
			bestInd_tmp = i; 
		}
	}
	cout << "....................................Tracker: Track...............1" << endl;
	UpdateDebugImage(keptRects, m_bb, scores);

	cout << "....................................Tracker: Track...............2" << endl;
	UpdateDebugImage_tmp(keptRects, m_bb_tmp, scores_tmp);

	cout << "....................................Tracker: Track...............3" << endl;
	if (bestInd != -1)
	{
		m_bb = keptRects[bestInd];
		m_bb_tmp = keptRects[bestInd_tmp];

	cout << "...................................Tracker: Track................4" << endl;
		UpdateLearner(image);
	cout << "...................................Tracker: Track................5" << endl;
#if VERBOSE		
		cout << "track score: " << bestScore << endl;
#endif
	}
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, 
	const FloatRect& centre, const vector<double>& scores)
{
	cout << ".................................UpdateDebugImage: samples/scores: " << scores.size() << "/" << samples.size() << endl;
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();
	m_debugImage.setTo(0);
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void Tracker::UpdateDebugImage_tmp(const vector<FloatRect>& samples, 
	const FloatRect& centre, const vector<double>& scores_tmp)
{
	cout << ".................................UpdateDebugImage_tmp: samples/scores: " << scores_tmp.size() << "/" << samples.size() << endl;
	//cout << "........Tracker::UpdateDebugImage_tmp->" << endl;
	double mn = VectorXd::Map(&scores_tmp[0], scores_tmp.size()).minCoeff();
	double mx = VectorXd::Map(&scores_tmp[0], scores_tmp.size()).maxCoeff();
	//cout << "........->" << mn << "/" << mx << endl;
	m_debugImage_tmp.setTo(0);
	//cout << "@@m_debugImage_tmp" << endl;
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		//cout << "	...x/y: " << x << "/" << y << endl;
		m_debugImage_tmp.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores_tmp[i]-mn)/(mx-mn));
		//cout << "	...,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,," << endl;
	}
}

void Tracker::Debug()
{
	cout << "Tracker::Debug" << endl;
	imshow("tracker", m_debugImage);
	cout << "	->screening" << endl;
	imshow("tracker_screening", m_debugImage_tmp);
	m_pLearner->Debug();
	cout << "	->END." << endl;
}

void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16);
	// vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);
	cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << keptRects.size() << " samples" << endl;
	m_pLearner->Update(sample, 0);
	cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << keptRects.size() << " samples" << endl;
}
