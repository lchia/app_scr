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

#include <iostream>
#include <fstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "vot.hpp"

using namespace std;
using namespace cv;

static const int kLiveBoxWidth = 80;
static const int kLiveBoxHeight = 80;

ofstream trainingLogFile;

void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

int main(int argc, char* argv[])
{
	// read config file
	string configPath = "config.txt";
	if (argc > 1)
	{
		configPath = argv[1];
	}
	Config conf(configPath);
	cout << conf << endl;
	cout << "conf.features.size() = " << conf.features.size() << endl;
	if (conf.features.size() == 0)
	{
		cout << "error: no features specified in config" << endl;
		return EXIT_FAILURE;
	}
	
	ofstream outFile;
	if (conf.resultsPath != "")
	{
		outFile.open(conf.resultsPath.c_str(), ios::out);
		if (!outFile)
		{
			cout << "error: could not open results file: " << conf.resultsPath << endl;
			return EXIT_FAILURE;
		}
	}
	
	//ofstream trainingLogFile;
	if (conf.trainingLogPath!= "")
	{
		trainingLogFile.open(conf.trainingLogPath.c_str(), ios::out);
		if (!trainingLogFile)
		{
			cout << "error: could not open training log file: " << conf.trainingLogPath << endl;
			return EXIT_FAILURE;
		}
	}

	if (trainingLogFile)
	{
		trainingLogFile << "=>main.cpp: read config file" << endl;
		trainingLogFile << conf << endl;
	}

	// if no sequence specified then use the camera
	bool useCamera = (conf.sequenceName == "");
	
	VideoCapture cap;
	
	int startFrame = -1;
	int endFrame = -1;
	FloatRect initBB;
	string imgFormat;
	float scaleW = 1.f;
	float scaleH = 1.f;
	
	if (useCamera)
	{
		if (!cap.open(0))
		{
			cout << "error: could not start camera capture" << endl;
			return EXIT_FAILURE;
		}
		startFrame = 0;
		endFrame = INT_MAX;
		Mat tmp;
		cap >> tmp;
		scaleW = (float)conf.frameWidth/tmp.cols;
		scaleH = (float)conf.frameHeight/tmp.rows;

		initBB = IntRect(conf.frameWidth/2-kLiveBoxWidth/2, conf.frameHeight/2-kLiveBoxHeight/2, kLiveBoxWidth, kLiveBoxHeight);
		cout << "press 'i' to initialise tracker" << endl;
	}
	else
	{
		// parse frames file
		string framesFilePath = conf.sequenceBasePath+"/"+conf.sequenceName+"/"+conf.sequenceName+"_frames.txt";
		ifstream framesFile(framesFilePath.c_str(), ios::in);
		if (!framesFile)
		{
			cout << "error: could not open sequence frames file: " << framesFilePath << endl;
			return EXIT_FAILURE;
		}
		string framesLine;
		getline(framesFile, framesLine);
		sscanf(framesLine.c_str(), "%d,%d", &startFrame, &endFrame);
		if (framesFile.fail() || startFrame == -1 || endFrame == -1)
		{
			cout << "error: could not parse sequence frames file" << endl;
			return EXIT_FAILURE;
		}
		
		imgFormat = conf.sequenceBasePath+"/"+conf.sequenceName+"/imgs/img%05d.png";
		
		if (trainingLogFile)
		{
			trainingLogFile << "=>main.cpp: parse frames file " << endl;
			trainingLogFile << "dataset: " << conf.sequenceName << endl;
			trainingLogFile << "startFrame: " << startFrame << endl;
			trainingLogFile << "endFrame:   " << endFrame << endl;
			trainingLogFile << "imgFormat:  " << imgFormat << endl;
		}

		// read first frame to get size
		char imgPath[256];
		sprintf(imgPath, imgFormat.c_str(), startFrame);
		Mat tmp = cv::imread(imgPath, 0);
		scaleW = (float)conf.frameWidth/tmp.cols;
		scaleH = (float)conf.frameHeight/tmp.rows;

		if (trainingLogFile)
		{
			trainingLogFile << endl;
			trainingLogFile << "=>main.cpp: read first frame to get size" << endl;
			trainingLogFile << "firstFrame: " << imgPath<< endl;
			trainingLogFile << "imgRows: " << tmp.rows << endl;
			trainingLogFile << "imgCols: " << tmp.cols << endl;
			trainingLogFile << "scaleW:   " << scaleW << endl;
			trainingLogFile << "scaleH:  " << scaleH << endl;
		}
		
		// read init box from ground truth file
		string gtFilePath = conf.sequenceBasePath+"/"+conf.sequenceName+"/"+conf.sequenceName+"_gt.txt";
		ifstream gtFile(gtFilePath.c_str(), ios::in);
		if (!gtFile)
		{
			cout << "error: could not open sequence gt file: " << gtFilePath << endl;
			return EXIT_FAILURE;
		}
		string gtLine;
		getline(gtFile, gtLine);
		float xmin = -1.f;
		float ymin = -1.f;
		float width = -1.f;
		float height = -1.f;
		sscanf(gtLine.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &width, &height);
		if (gtFile.fail() || xmin < 0.f || ymin < 0.f || width < 0.f || height < 0.f)
		{
			cout << "error: could not parse sequence gt file" << endl;
			return EXIT_FAILURE;
		}
		initBB = FloatRect(xmin*scaleW, ymin*scaleH, width*scaleW, height*scaleH);

		if (trainingLogFile)
		{
			trainingLogFile << endl;
			trainingLogFile << "=>main.cpp: read init box from ground truth file" << endl;
			trainingLogFile << "ground truth file name: " << conf.sequenceName+"_gt.txt" << endl;
			trainingLogFile << "gtLine: " << gtLine << endl;
			trainingLogFile << "initBB:  " << initBB << endl;
		}
	}
	
	
	
	Tracker tracker(conf);
	if (!conf.quietMode)
	{
		namedWindow("result");
	}
	
	Mat result(conf.frameHeight, conf.frameWidth, CV_8UC3);
	bool paused = false;
	bool doInitialise = false;
	srand(conf.seed);

	if (trainingLogFile)
	{
		trainingLogFile << "=>frameLoop: first frame" << endl;
	}
	for (int frameInd = startFrame; frameInd <= endFrame; ++frameInd)
	//for (int frameInd = startFrame; frameInd <= startFrame+1;  ++frameInd)
	{
		Mat frame;
		if (trainingLogFile)
		{
			trainingLogFile << endl;
			trainingLogFile << "=>main.cpp: start frame loop" << endl;
		}

		if (useCamera)
		{
			Mat frameOrig;
			cap >> frameOrig;
			resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
			flip(frame, frame, 1);
			frame.copyTo(result);
			if (doInitialise)
			{
				if (tracker.IsInitialised())
				{
					tracker.Reset();
				}
				else
				{
					tracker.Initialise(frame, initBB);
				}
				doInitialise = false;
			}
			else if (!tracker.IsInitialised())
			{
				rectangle(result, initBB, CV_RGB(255, 255, 255));
			}
		}
		else
		{			
			char imgPath[256];
			sprintf(imgPath, imgFormat.c_str(), frameInd);
			Mat frameOrig = cv::imread(imgPath, 0);
			if (frameOrig.empty())
			{
				cout << "error: could not read frame: " << imgPath << endl;
				return EXIT_FAILURE;
			}
			resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
			cvtColor(frame, result, CV_GRAY2RGB);
		
			if (trainingLogFile)
			{
				trainingLogFile << ">>frameLoop: " << frameInd << "-th " << endl;
				trainingLogFile << "  frame name: " << imgPath << endl;
				trainingLogFile << "  origImage size: " << frameOrig.size() << endl;
				//trainingLogFile << "frameOrig rows: " << frameOrig.rows << endl;
				//trainingLogFile << "frameOrig cols: " << frameOrig.cols << endl;
				trainingLogFile << "  frame size: " << frame.size() << endl;
				//trainingLogFile << "frame rows: " << frame.rows << endl;
				//trainingLogFile << "frame cols: " << frame.cols << endl;
			}

			if (frameInd == startFrame)
			{
				tracker.Initialise(frame, initBB);
			}

		}
		
		if (tracker.IsInitialised())
		{
			if (trainingLogFile) {
				trainingLogFile << "@@@Gate of tracking for each frame: main->tracker.Track(frame) .." << std::endl;
			}
			tracker.Track(frame);
			
			if (!conf.quietMode && conf.debugMode)
			{
				tracker.Debug();
			}
			
			rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));
			
			if (outFile)
			{
				const FloatRect& bb = tracker.GetBB();
				outFile << bb.XMin()/scaleW << "," << bb.YMin()/scaleH << "," << bb.Width()/scaleW << "," << bb.Height()/scaleH << endl;
			}
		}
		
		if (!conf.quietMode)
		{
			imshow("result", result);
			int key = waitKey(paused ? 0 : 1);
			if (key != -1)
			{
				if (key == 27 || key == 113) // esc q
				{
					break;
				}
				else if (key == 112) // p
				{
					paused = !paused;
				}
				else if (key == 105 && useCamera)
				{
					doInitialise = true;
				}
			}
			if (conf.debugMode && frameInd == endFrame)
			{
				cout << "\n\nend of sequence, press any key to exit" << endl;
				waitKey();
			}
		}
	}
	
	if (outFile.is_open())
	{
		outFile.close();
	}

	if (trainingLogFile.is_open())
	{
		trainingLogFile.close();
	}
	
	return EXIT_SUCCESS;
}
