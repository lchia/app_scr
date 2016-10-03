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

#include "RawFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace Eigen;
using namespace cv;
using namespace std;

static const int kPatchSize = 16;

RawFeatures::RawFeatures(const Config& conf) :
	m_patchImage(kPatchSize, kPatchSize, CV_8UC1)
{
	SetCount(kPatchSize*kPatchSize);
}

void RawFeatures::UpdateFeatureVector(const Sample& s)
{
	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................0" << endl;
	IntRect rect = s.GetROI(); // note this truncates to integers

	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................1" << endl;
	cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................2" << endl;
	cv::resize(s.GetImage().GetImage(0)(roi), m_patchImage, m_patchImage.size());
	//equalizeHist(m_patchImage, m_patchImage);
	
	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................3" << endl;
	int ind = 0;
	for (int i = 0; i < kPatchSize; ++i)
	{
		uchar* pixel = m_patchImage.ptr(i);
	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................4" << endl;
		for (int j = 0; j < kPatchSize; ++j, ++pixel, ++ind)
		{
	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................5" << endl;
			m_featVec[ind] = ((double)*pixel)/255;
	cout << "RawFeatures::UpdateFeatureVector(const Sample& s).....................6" << endl;
		}
	}
}
