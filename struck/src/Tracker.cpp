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
        std::cout<<"m_config: "<<m_config<<std::endl;//added by lch
        std::cout<<"numFeatures: "<<numFeatures<<std::endl;//added by lch
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
	std::cout << "@@@@@@@@@@@@@@@@@@@@@Tracker::Initialise------------" << std::endl;
	if (trainingLogFile)
        {
        	trainingLogFile << "  @Tracker::Initialise" << endl;
        }

	m_bb = IntRect(bb);
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);

	if (trainingLogFile)
        {
        	trainingLogFile << "   m_bb: " << m_bb << endl;
        	trainingLogFile << "   ImageRep: image(frame, m_needsIntegralImage, m_needsIntegralHist)" << std::endl;
        }

	for (int i = 0; i < 1; ++i)
	{
		if (trainingLogFile)
        	{
        		trainingLogFile << "  @UpdateLearner(image)" << endl;
        	}
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

        std::cout<<"rects: "<<rects.size()<<std::endl;//added by lch 
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}
	
	MultiSample sample(image, keptRects);
	
	//This is the Gate of Eval "scores" of each rects given the "sample"
	if(trainingLogFile) {
		trainingLogFile << "===========================Gate of Eval(sample, scroes)" << std::endl;
	}
	vector<double> scores;
	m_pLearner->Eval(sample, scores);
        
	
	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
                if (i%500 == 0)
                        std::cout<<"scores[i]: "<< scores[i] <<std::endl;//added by lch
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i;
		}
	}
	
	UpdateDebugImage(keptRects, m_bb, scores);
	
	if(trainingLogFile) {
		trainingLogFile << "===========================Got the predict box" << std::endl;
		trainingLogFile << "	>bestScore: " << bestScore << std::endl;
		trainingLogFile << "	>bestInd: " << bestInd << std::endl;
		trainingLogFile << "	>m_bb: " << keptRects[bestInd] << std::endl;
		trainingLogFile << "	>UpdateLearner(image) after got the 'm_bb'." << std::endl;
	}
	if (bestInd != -1)
	{
		m_bb = keptRects[bestInd];
		UpdateLearner(image);
#if VERBOSE		
		cout << "track score: " << bestScore << endl;
#endif
	}
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, const FloatRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();
	m_debugImage.setTo(0);
	for (int i = 0; i < (int)samples.size(); ++i)
	{
                if (i%1000==0)
                       std::cout<<"UpdateDebugImage: "<< i <<std::endl;//added by lch
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void Tracker::Debug()
{
	imshow("tracker", m_debugImage);
	m_pLearner->Debug();
}

void Tracker::UpdateLearner(const ImageRep& image)
{
	if (trainingLogFile)
        {
        	trainingLogFile << "  @Tracker::UpdateLearner .." << endl;
        	trainingLogFile << "    Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16) -> rects" << endl;
        }
	// note these return the centre sample at index 0
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}

	if (trainingLogFile)
        {
        	
        	trainingLogFile << "    centre sample: rects[0] = " << rects[0] << endl;
        	trainingLogFile << "    number of rects: rects.size() = " << rects.size() << endl;
        	trainingLogFile << "    number of keptRects: keptRects.size() = " << keptRects.size() << std::endl;
        }
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);

	if (trainingLogFile)
        {
        	
        	trainingLogFile << "    @MultiSameple sample(image, keptRects) .."<< endl;
		trainingLogFile << "    @m_pLearner->Update(smaple, 0) .." << endl;
        }
	m_pLearner->Update(sample, 0);
}
