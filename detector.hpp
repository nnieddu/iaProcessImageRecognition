#pragma once
#include <string>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

#include <Windows.h>

#include "defines.h"

#include "aimbot.h" //////////

class detector
{
public:
	detector(int width, int height);
	~detector(){};

	void detectYolo(cv::Mat &image);

	cv::dnn::Net m_net;
	std::vector<std::string> m_classes;

	void start(cv::Mat& image); ////////////////
private:
	int width;
	int height;

	struct Detection
	{
		int class_id;
		float confidence;
		cv::Rect box;
	};

	//// non-maximum suppression (NMS) algorithm
	const float INPUT_WIDTH = 640.0;
	const float INPUT_HEIGHT = 640.0;
	// const float INPUT_WIDTH = 416.0;
	// const float INPUT_HEIGHT = 416.0;

	const float SCORE_THRESHOLD = 0.2;
	const float NMS_THRESHOLD = 0.4;
	const float CONFIDENCE_THRESHOLD = 0.2;

	void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
	cv::Mat format_yolo(const cv::Mat &source);




	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
	void draw_box(float conf, int left, int top, int right, int bottom, cv::Mat& frame);
	std::vector<cv::String> get_outputs_names(const cv::dnn::Net& net);

	float m_confidence = 0.5f;
	float m_threshold = 0.35f;
	int m_activation_range = 125;
};
