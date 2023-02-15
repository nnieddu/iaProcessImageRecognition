#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

#include <string>
#include <vector>
#include <fstream>

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
<<<<<<< HEAD
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
=======
float m_confidence = 0.5f;
	float m_threshold = 0.35f;
	int m_activation_range = 125;
	int width;
	int height;

	void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs);
	void draw_box(float conf, int left, int top, int right, int bottom, cv::Mat &frame);
	std::vector<cv::String> get_outputs_names(const cv::dnn::Net& net);


	void detectYoloV3(cv::Mat &image);
	void detectYoloV5(cv::Mat &image);
	void detectYoloV7(cv::Mat &image);
>>>>>>> b783de7ab1eaf3007e3ae5c21fb23ec479544062

	const float SCORE_THRESHOLD = 0.2;
	const float NMS_THRESHOLD = 0.4;
	const float CONFIDENCE_THRESHOLD = 0.4;

	void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
	cv::Mat format_yolov5(const cv::Mat &source);
};