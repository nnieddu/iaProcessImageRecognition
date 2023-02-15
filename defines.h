#pragma once

// #define MODEL_FOLDER "..\\..\\IAmodels\\yolo-v3-tiny\\"
// #define LABELS_FILE_NAME "classes.txt"
// // #define LABELS_FILE_NAME "coco-dataset.labels"
// #define YOLO_CFG_FILE_NAME "yolov3-tiny.cfg"
// #define YOLO_WEIGHTS_FILE_NAME "yolov3-tiny.weights"

#define MODEL_FOLDER "..\\..\\IAmodels\\yolo-v5\\"
#define LABELS_FILE_NAME "classes.txt"
#define YOLO_CFG_FILE_NAME ""
#define YOLO_WEIGHTS_FILE_NAME "yolov5n.onnx"

// #define MODEL_FOLDER "..\\..\\IAmodels\\yolo-v7\\"
// // #define LABELS_FILE_NAME "coco_classes.txt"
// #define LABELS_FILE_NAME "classes.txt"
// #define YOLO_CFG_FILE_NAME ""
// #define YOLO_WEIGHTS_FILE_NAME "yolov7-tiny.onnx"




	// for (int i = 0; i < rows; ++i, data += dimensions)
	// {
	// 	float confidence = data[4];
	// 	if (confidence >= CONFIDENCE_THRESHOLD)
	// 	{
	// 		float *classes_scores = data + 5;
	// 		cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
	// 		cv::Point class_id;
	// 		double max_class_score;
	// 		minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
	// 		if (max_class_score > SCORE_THRESHOLD)
	// 		{
	// 			confidences.push_back(confidence);
	// 			class_ids.push_back(class_id.x);

	// 			float x = data[0];
	// 			float y = data[1];
	// 			float w = data[2];
	// 			float h = data[3];
	// 			int left = int((x - 0.5 * w) * x_factor);
	// 			int top = int((y - 0.5 * h) * y_factor);
	// 			int width = int(w * x_factor);
	// 			int height = int(h * y_factor);
	// 			boxes.push_back(cv::Rect(left, top, width, height));
	// 		}
	// 	}
	// }

// for (const auto &output : outputs) {
// 		for (int i = 0; i < output.rows; i++) {
// 				cv::Mat detection = output.row(i);

// 				// Extract the class ID and confidence
// 				cv::Mat scores = detection.colRange(5, output.cols);
// 				cv::Point class_id;
// 				double confidence;
// 				cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id);

// 				// Check if the detection exceeds the confidence threshold
// 				if (confidence >= CONFIDENCE_THRESHOLD) {
// 						// Extract the bounding box coordinates
// 						int center_x = static_cast<int>(detection.at<float>(0) * image.cols);
// 						int center_y = static_cast<int>(detection.at<float>(1) * image.rows);
// 						int width = static_cast<int>(detection.at<float>(2) * image.cols);
// 						int height = static_cast<int>(detection.at<float>(3) * image.rows);
// 						int left = center_x - width / 2;
// 						int top = center_y - height / 2;

// 						// Add the detection to the output vector
// 						class_ids.push_back(class_id.x);
// 						confidences.push_back(static_cast<float>(confidence));
// 						boxes.emplace_back(left, top, width, height);
// 				}
// 		}
// }
