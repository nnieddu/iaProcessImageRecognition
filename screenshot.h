#pragma once
#include <opencv2/opencv.hpp>
#include <windows.h>

class screenshot {
public:
	screenshot(std::string proccessName);
	~screenshot();
	cv::Mat& get();

	void getAndSetProcessScreenSize(HWND hwnd);

	int getHeight() { return m_height; }
	int getWidth() { return m_width; }

private:
	HDC m_hWDC;
	HDC m_hScreen;
	HBITMAP m_hBitmap;
	BITMAPINFO m_bitmapinfo;

	int m_width;
	int m_height;

	int m_left;
	int m_top;

	char* m_data;
	cv::Mat* m_screen;

	HGDIOBJ m_hGDI_temp;

	bool processFound;
};