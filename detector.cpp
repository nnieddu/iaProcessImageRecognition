#include "detector.h"

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

cv::Mat detector::format_yolov5(const cv::Mat &source)
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

	auto input_image = format_yolov5(image);

	cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
				
	net.setInput(blob);
	std::vector<cv::Mat> outputs;

	net.forward(outputs, net.getUnconnectedOutLayersNames()); ////// PERF DOWN
	
	float x_factor = input_image.cols / INPUT_WIDTH;
	float y_factor = input_image.rows / INPUT_HEIGHT;

	float *data = (float *)outputs[0].data;

	const int dimensions = 85;
	const int rows = 25200;

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < rows; ++i)
	{

		float confidence = data[4];
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float *classes_scores = data + 5;
			cv::Mat scores(1, className.size(), CV_32FC1, classes_scores); ///////////
			cv::Point class_id;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			if (max_class_score > SCORE_THRESHOLD)
			{
				confidences.push_back(confidence);

			// 	float x = data[0];
			// 	float y = data[1];
			// 	float w = data[2];
			// 	float h = data[3];
			// 	int left = int((x - 0.5 * w) * x_factor);
			// 	int top = int((y - 0.5 * h) * y_factor);
			// 	int width = int(w * x_factor);
			// 	int height = int(h * y_factor);
			// 	boxes.push_back(cv::Rect(left, top, width, height));
			// }
		}

		data += 85;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
	for (int i = 0; i < nms_result.size(); i++)
	{
		int idx = nms_result[i];
		Detection result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
}
}

void detector::detectYolo(cv::Mat &image)
{
	std::vector<Detection> output;
	this->detect(image, m_net, output, m_classes);
	size_t detections = output.size();
	for (int i = 0; i < detections; ++i)
	{
		auto detection = output[i];
		auto box = detection.box;
		auto classId = detection.class_id;

		const auto color = colors[classId % colors.size()];

		cv::rectangle(image, box, colors[classId % colors.size()], 3);
		cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), colors[classId % colors.size()], cv::FILLED);
		cv::putText(image, std::to_string(detection.confidence) + "  " + m_classes[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
}

detector::detector(int width, int height) : INPUT_WIDTH(640.0), INPUT_HEIGHT(640.0)
{
	this->width = width;
	this->height = height;

	// const float INPUT_WIDTH = 416.0; //yolo3 tiny
	// const float INPUT_HEIGHT = 416.0; // yolo3 tiny

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
		
	if (cv::cuda::getCudaEnabledDeviceCount() > 0)
	{
		std::cout << "=== CUDA TARGET ===" << std::endl;
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	else 
	if (cv::ocl::haveOpenCL())
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