#include "detector.hpp"
#include "defines.h"

#include <Windows.h>

#include <math.h>

// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //
void detector::draw_box(float conf, int left, int top, int right, int bottom, cv::Mat& frame) {
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
    std::string label = cv::format("%.2f", conf);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
}

std::vector<cv::String> detector::get_outputs_names(const cv::dnn::Net& net) {
    static std::vector<cv::String> names;
    if (names.empty()) {
        std::vector<int> outLayers =
            net.getUnconnectedOutLayers();
        std::vector<cv::String> layersNames =
            net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void detector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<int> classes_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > m_confidence) {
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

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        draw_box(confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);

				std::cout << "DETECT ! " << std::endl;
    }

    return;
}

static clock_t current_ticks, delta_ticks;
static clock_t fps = 0;

void detector::start(cv::Mat& image) {
    current_ticks = clock();

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1 / 255.0,
        cv::Size(m_activation_range, m_activation_range), cv::Scalar(0, 0, 0), true, false);
    m_net.setInput(blob);
    std::vector<cv::Mat> outs;
    m_net.forward(outs, get_outputs_names(m_net));

    postprocess(image, outs);

    std::string label = cv::format("FPS: %u", (unsigned int)fps);
    cv::putText(image, label, cv::Point(0, 15),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    cv::Mat detected_frame;
    image.convertTo(detected_frame, CV_8U);
    // cv::imshow("NN", detected_frame);
		image = detected_frame;
    // number of processed frames per sec.
    delta_ticks = clock() - current_ticks;
    if (delta_ticks > 0)
        fps = CLOCKS_PER_SEC / delta_ticks;

    return;
}
// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //

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

	// auto input_image = format_yolo(image);
	// cv::dnn::blobFromImage(input_image, blob, 1 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);

	auto input_image = image;
	cv::Mat resized;
	cv::resize(image, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
	cv::dnn::blobFromImage(resized, blob, 1 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);


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
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);

				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Apply non-maximum suppression to remove overlapping boxes
	std::vector<int> nms_indices;
	cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);
	// output.clear();
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

	// m_net = cv::dnn::readNet(ModelPath);
	m_net = cv::dnn::readNetFromDarknet(ConfigPath, ModelPath);
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