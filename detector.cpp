#include "detector.hpp"
#include "defines.h"
#include <Windows.h>

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

cv::Mat detector::format_yolo(const cv::Mat &source)
{
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);
	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}

void detector::detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className)
{
	cv::Mat blob;
	auto input_image = format_yolo(image);
	cv::dnn::blobFromImage(input_image, blob, 1 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
	net.setInput(blob);
	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	float x_factor = input_image.cols / INPUT_WIDTH;
	float y_factor = input_image.rows / INPUT_HEIGHT;
	float *data = (float *)outputs[0].data;
	const int rows = 25200;
	const int dimensions = 85;

	for (int i = 0; i < rows; ++i, data += dimensions)
	{
		float confidence = data[4];
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float *classes_scores = data + 5;
			cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
			cv::Point class_id;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			if (max_class_score > SCORE_THRESHOLD)
			{
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);

				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Apply non-maximum suppression to remove overlapping boxes
	std::vector<int> nms_indices;
	cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);
	// output.clear();
	// for (auto idx : nms_indices)
	// {
	for (int i = 0; i < nms_indices.size(); i++)
	{
		int idx = nms_indices[i];
		Detection detection;
		detection.class_id = class_ids[idx];
		detection.confidence = confidences[idx];
		detection.box = boxes[idx];
		output.push_back(detection);
	}
}

void detector::detectYolo(cv::Mat &image)
{
	std::vector<Detection> output;
	detect(image, m_net, output, m_classes);
	size_t detections = output.size();
	for (int i = 0; i < detections; ++i)
	{
		auto detection = output[i];
		auto box = detection.box;
		auto classId = detection.class_id;

		const auto color = colors[classId % colors.size()];

		cv::rectangle(image, box, color, 3);
		cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
		cv::putText(image, std::to_string(detection.confidence) + "  " + m_classes[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
}

detector::detector(int width, int height)
{
	this->width = width;
	this->height = height;

	char cwd[MAX_PATH];
	GetCurrentDirectoryA(MAX_PATH, cwd);
	const std::string str_exe_path = cwd;
	std::string ModelPath = str_exe_path + "\\" + MODEL_FOLDER + YOLO_WEIGHTS_FILE_NAME;
	std::string ConfigPath = str_exe_path + "\\" + MODEL_FOLDER + YOLO_CFG_FILE_NAME;
	std::string datasetLabelsPath = str_exe_path + "\\" + MODEL_FOLDER + LABELS_FILE_NAME;
	std::ifstream ifs(datasetLabelsPath);
	std::string line;
	while (getline(ifs, line))
		m_classes.push_back(line);
	std::cout << "DATASET LABEL SIZE = " << m_classes.size() << std::endl;

	m_net = cv::dnn::readNet(ModelPath);
	if (m_net.empty())
		std::cerr << "Can't load network" << std::endl;

	if (cv::cuda::getCudaEnabledDeviceCount() > 0)
	{
		std::cout << "=== CUDA TARGET ===" << std::endl;
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	else if (cv::ocl::haveOpenCL())
	{
		std::cout << "=== OPENCL TARGET ===" << std::endl;
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
		cv::ocl::setUseOpenCL(true);
		cv::ocl::useOpenCL();
	}
	else
	{
		std::cout << "=== CPU TARGET ===" << std::endl;
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
}