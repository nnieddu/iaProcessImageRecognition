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
#include "screenshot.hpp"
#include "detector.hpp"
#include "fpsCounter.hpp"

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

int main(int ac, char **av)
{
	std::cout << std::endl << std::endl << std::endl; ///// TESTING
	FpsCounter countFps;
	bool activate = true;
	bool exit = false;
	std::string processName;

	if (ac == 2 && av[1])
		processName = av[1];

	screenshot screen(processName);
	detector detectObj(screen.getWidth(), screen.getHeight());
	cv::Mat frame;

	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
	// cv::namedWindow("Result", cv::WINDOW_NORMAL);
	// cv::namedWindow("Result", cv::WINDOW_GUI_EXPANDED );
	// cv::setWindowProperty("Result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	int moyFps = 0; ///
	int nbrOfBoucle = 0; ///

	while (!exit)
	{
		disableOrActivate(activate, exit);
		while (activate)
		{
				countFps.setCurrentTick();

				nbrOfBoucle++;
				moyFps += countFps.get();
				
				disableOrActivate(activate, exit);
				frame = screen.get();
				// detectObj.detectYolo(frame);
				detectObj.start(frame);
				cv::putText(frame, std::to_string(countFps.get()), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				cv::imshow("Result", frame);
				cv::waitKey(1);
				countFps.update();
		}
	}
	cv::destroyAllWindows();
	std::cout << "FPS = " << moyFps / nbrOfBoucle << std::endl;
	std::cout << "CLEAN EXIT" << std::endl;
	return 0;
}