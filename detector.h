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

	const float INPUT_WIDTH;
	const float INPUT_HEIGHT;

	const float SCORE_THRESHOLD = 0.2;
	const float NMS_THRESHOLD = 0.4;
	const float CONFIDENCE_THRESHOLD = 0.4;

	void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
	cv::Mat format_yolov5(const cv::Mat &source);
};