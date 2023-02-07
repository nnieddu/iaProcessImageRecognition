#include <Windows.h>
#include <tchar.h>

#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "defines.h"
#include "screenshot.h"
#include "detector.h"
#include "fpsCounter.h"

void disableOrActivate(bool &activate, bool &exit)
{
	if (GetAsyncKeyState(109)) // num pad - key (substract key)
		activate = false;
	if (GetAsyncKeyState(107)) // num pad + key (addition key)
		activate = true;	// esc key -> exit 
	// ~ getWindowProperty return -1 when clicking windows cross 
	if (GetAsyncKeyState(27)) 
	{
		activate = false;
		exit = true;
	}
}


int main()
{
	FpsCounter countFps;
	bool activate = true;
	bool exit = false;
	screenshot screen(" ");

	detector detectObj(screen.getWidth(), screen.getHeight());
	cv::Mat frame;

	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
	// cv::namedWindow("Result", cv::WINDOW_NORMAL);
	// cv::namedWindow("Result", cv::WINDOW_GUI_EXPANDED );
	// cv::setWindowProperty("Result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	while (!exit)
	{
		disableOrActivate(activate, exit);
		while (activate)
		{
				countFps.setCurrentTick();
				disableOrActivate(activate, exit);
				frame = screen.get();
				detectObj.start(frame);
				cv::putText(frame, std::to_string(countFps.updateAndGet()), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				// std::cout << countFps.updateAndGet() << std::endl;
				cv::imshow("Result", frame);
				cv::waitKey(1);
		}
	}
	cv::destroyAllWindows();
	std::cout << "CLEAN EXIT" << std::endl;
	return 0;
}