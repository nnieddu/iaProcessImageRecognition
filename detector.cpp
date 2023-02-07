#include "detector.h"
#include "defines.h"
#include <Windows.h>

void detector::draw_box(float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
	std::string label = cv::format("%.2f", conf);
	putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
}

// std::vector<cv::String> detector::get_outputs_names(const cv::dnn::Net &net)
// {
// 	static std::vector<cv::String> names;
// 	if (names.empty())
// 	{
// 		std::vector<int> outLayers = net.getUnconnectedOutLayers();
// 		std::vector<cv::String> layersNames = net.getLayerNames();
// 		names.resize(outLayers.size());
// 		for (size_t i = 0; i < outLayers.size(); ++i)
// 			names[i] = layersNames[outLayers[i] - 1];
// 	}
// 	return names;
// }

void detector::postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
	std::vector<int> classes_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float *data = (float *)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point class_id_point;
			double confidence;
			cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
			if (confidence > m_confidence)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classes_ids.push_back(class_id_point.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// exclusion of overlapping boxes and other trash
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confidence, m_threshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		draw_box(confidences[idx], box.x, box.y,
						 box.x + box.width, box.y + box.height, frame);

		// todo: check if the aim is enabled
		// aimbot::aim_to(box.x, box.y, box.width, box.height);
	}
}

void detector::detectYoloV3(cv::Mat &image)
{
	cv::Mat blob;
	cv::dnn::blobFromImage(image, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false);

	// Set the input to the network
	m_net.setInput(blob);

	// Run forward pass
	cv::Mat output = m_net.forward();

	// Get the confidence scores and bounding box locations
	cv::Mat scores = output.row(0).colRange(5, output.cols);
	cv::Mat boxes = output.row(0).colRange(0, 4);
	cv::Mat classes = output.row(0).colRange(5, output.cols);

	// Perform non-maximum suppression to remove overlapping boxes
	cv::Mat indices;
	// cv::dnn::NMSBoxes(boxes, scores, 0.5, 0.4, indices);

	// Draw bounding boxes around the objects
	for (int i = 0; i < indices.rows; ++i)
	{
		int idx = indices.at<int>(i, 0);
		float confidence = scores.at<float>(idx, 0);

		if (confidence > 0.5)
		{
			float x = boxes.at<float>(idx, 0);
			float y = boxes.at<float>(idx, 1);
			float w = boxes.at<float>(idx, 2);
			float h = boxes.at<float>(idx, 3);
			int classId = classes.at<float>(idx, 0);

			cv::Rect object((int)(x * image.cols), (int)(y * image.rows),
											(int)(w * image.cols), (int)(h * image.rows));
			rectangle(image, object, cv::Scalar(0, 255, 0), 2);

			// Display object label
			cv::String label = cv::format("%.2f", confidence);
			putText(image, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		}
	}
}




const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;
// const float CONFIDENCE_THRESHOLD = 0.75;

struct Detection
{
	int class_id;
	float confidence;
	cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source)
{
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);
	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className)
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

void detector::detectYoloV5(cv::Mat &image)
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

		if (m_classes[classId] == "person")
		{
			cv::rectangle(image, box, colors[classId % colors.size()], 3);
			cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), colors[classId % colors.size()], cv::FILLED);
			cv::putText(image, std::to_string(detection.confidence) + "  " + m_classes[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}
}

void detector::start(cv::Mat &image)
{
	if (yolov3)
	{
		// std::cout << "=== detectYoloV3 ===" << std::endl;
		detectYoloV3(image);
	}
	else
	{
		// std::cout << "=== detectYoloV5 ===" << std::endl;
		detectYoloV5(image);
	}

}

detector::detector(int width, int height)
{
	width = width;
	height = height;
	char cwd[MAX_PATH];
	GetCurrentDirectoryA(MAX_PATH, cwd);
	const std::string str_exe_path = cwd;

	std::string ModelPath = str_exe_path + "\\" + MODEL_FOLDER + YOLO_WEIGHTS_FILE_NAME;
	std::string ConfigPath = str_exe_path + "\\" + MODEL_FOLDER + YOLO_CFG_FILE_NAME;
	std::string datasetLabelsPath = str_exe_path + "\\" + MODEL_FOLDER + LABELS_FILE_NAME;

	// std::vector<std::string> datasetLabels;
	std::ifstream ifs(datasetLabelsPath);
	std::string line;
	while (getline(ifs, line))
		m_classes.push_back(line);
	std::cout << "DATASET LABEL SIZE = " << m_classes.size() << std::endl;
	// -----------------------------
	if (YOLO_CFG_FILE_NAME == "")
	{
		m_net = cv::dnn::readNet(ModelPath);
		yolov3 = false;
		std::cout << "=== YOLOv5 LOADED ===" << std::endl;
	}
	else
	{
		m_net = cv::dnn::readNet(ModelPath, ConfigPath);
		yolov3 = true;
		std::cout << "=== YOLOv3 LOADED ===" << std::endl;
	}
	if (cv::cuda::getCudaEnabledDeviceCount() > 0)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	else 
	if 
	(cv::ocl::haveOpenCL())
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
		cv::ocl::setUseOpenCL(true);
		cv::ocl::useOpenCL();
	}
	else
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
}