#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <string>
#include <vector>

#include <fstream>
// #include "screenshot.h"

class detector
{
public:
	detector(int width, int height);
	~detector(){};
	void start(cv::Mat &image);

	// screenshot screen;
	bool yolov3;
	cv::dnn::Net m_net;
	std::vector<std::string> m_classes;
private:
	float m_confidence = 0.5f;
	float m_threshold = 0.35f;
	int width;
	int height;

	void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs);
	void draw_box(float conf, int left, int top, int right, int bottom, cv::Mat &frame);

	void detectYoloV3(cv::Mat &image);
	void detectYoloV5(cv::Mat &image);

	// std::vector<cv::String> get_outputs_names(const cv::dnn::Net &net);
};