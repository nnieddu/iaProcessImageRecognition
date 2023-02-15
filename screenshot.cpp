#include "screenshot.hpp"

#include <windows.h>
#include <tchar.h>
#include <vector>

screenshot::screenshot(std::string proccessName)
{
	m_left = 0;
	m_top = 0;
	HWND hwnd;
	if (!proccessName.empty())
	{
		hwnd = FindWindow(NULL, proccessName.c_str());
		std::cout << "=== CAPTURING PROCESS : [" << proccessName << "] ===" << std::endl;
	}
	else
	{
		std::cout << "=== CAPTURING MAIN SCREEN ===" << std::endl;
		hwnd = GetDesktopWindow();
	}

	// Get the device context (DC) of the window handle
	m_hWDC = GetWindowDC(hwnd);
	// Create a compatible device context (DC) with the window DC
	m_hScreen = CreateCompatibleDC(m_hWDC);
	// Get the client rectangle of the window
	RECT rcClient;
	GetClientRect(hwnd, &rcClient);
	m_width += rcClient.right - rcClient.left;
	m_height += rcClient.bottom - rcClient.top;

	m_width /= 2;	 ////// TESTING
	m_height /= 2; ///// TESTING
	std::cout << m_width << " " << m_height << std::endl; ///// TESTING

	// Create a bitmap compatible with the window DC, with the size of the client rectangle
	m_hBitmap = CreateCompatibleBitmap(m_hWDC, m_width, m_height);
	// Select the bitmap into the compatible DC
	m_hGDI_temp = SelectObject(m_hScreen, m_hBitmap);

	// Init bitmap info for all futur frames
	BITMAPINFOHEADER bi = {0};
	m_bitmapinfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	m_bitmapinfo.bmiHeader.biWidth = m_width;
	m_bitmapinfo.bmiHeader.biHeight = -m_height;
	m_bitmapinfo.bmiHeader.biPlanes = 1;
	m_bitmapinfo.bmiHeader.biBitCount = 24;
	m_bitmapinfo.bmiHeader.biCompression = BI_RGB;

	int step = static_cast<int>(ceil(m_width * 3 / static_cast<double>(4))) * 4;
	m_data = new char[step * m_height];
	m_screen = new cv::Mat(m_height, m_width, CV_8UC3, m_data, step);
}

screenshot::~screenshot()
{
	SelectObject(m_hScreen, m_hGDI_temp);
	DeleteObject(m_hBitmap);
	DeleteDC(m_hScreen);
	delete m_screen;
	delete[] m_data;
}

cv::Mat &screenshot::get()
{
	// BitBlt (Bit Block Transfer) copies the contents of one DC to another
	BitBlt(m_hScreen, 0, 0, m_width, m_height, m_hWDC, m_left, m_top, SRCCOPY);
	GetDIBits(m_hScreen, m_hBitmap, 0, m_height, m_data, &m_bitmapinfo, DIB_RGB_COLORS);
	return *m_screen;
}
