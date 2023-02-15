#pragma once
#include <string>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

#include <Windows.h>

#include "defines.h"

class detector
{
public:
	detector(int width, int height);
	~detector(){};

	void detectYolo(cv::Mat &image);

	cv::dnn::Net m_net;
	std::vector<std::string> m_classes;

private:
	int width;
	int height;

	struct Detection
	{
		int class_id;
		float confidence;
		cv::Rect box;
	};

	// non-maximum suppression (NMS) algorithm
	const float INPUT_WIDTH = 640.0;
	const float INPUT_HEIGHT = 640.0;

	const float SCORE_THRESHOLD = 0.2;
	const float NMS_THRESHOLD = 0.4;
	const float CONFIDENCE_THRESHOLD = 0.2;

	void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
	cv::Mat format_yolo(const cv::Mat &source);
};