#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <thread>

using namespace cv;
using namespace std;


int main()
{
	Mat bgr(1080, 1920, CV_8UC4);
	bgr = imread("color.png");
	Mat depth(512, 512, CV_16UC1);
	depth = imread("depth.png", IMREAD_ANYDEPTH); 
	// 3. display
	thread th = std::thread([&] {
		while (true)
		{
			namedWindow("Display window1", WINDOW_AUTOSIZE);
			imshow("rgb_image", bgr);
			waitKey(1);
			namedWindow("Display window2", WINDOW_AUTOSIZE);
			imshow("depth_image", depth * 20);
			waitKey(1);
		}
		});

	Eigen::Matrix3f K_ir;           // ir intrinx para matrix
	K_ir <<
		368.8057, 0, 255.5000,
		0, 369.5268, 211.5000,
		0, 0, 1;
	Eigen::Matrix3f K_rgb;          // rgb intrinx para matrix
	K_rgb <<
		1044.7786, 0, 985.9435,
		0, 1047.2506, 522.7765,
		0, 0, 1;

	Eigen::Matrix3f R_ir2rgb;
	Eigen::Matrix3f R;
	Eigen::Vector3f T_temp;
	Eigen::Vector3f T;
	R_ir2rgb <<
		0.9996, 0.0023, -0.0269,
		-0.0018, 0.9998, 0.0162,
		0.0269, -0.0162, 0.9995;
	T_temp <<
		65.9080,
		-4.1045,
		-13.9045;
	R = K_rgb * R_ir2rgb * K_ir.inverse();
	T = K_rgb * T_temp;


	//projection calculation
	Mat result(512, 512, CV_8UC3);
	int i = 0;
	for (int row = 0; row < 512; row++)
	{
		for (int col = 0; col < 512; col++)
		{
			unsigned short* p = (unsigned short*)depth.data;
			unsigned short depthValue = p[row * 512 + col];
			//cout << "depthValue       " << depthValue << endl;
			if (depthValue != -std::numeric_limits<unsigned short>::infinity() && depthValue != -std::numeric_limits<unsigned short>::infinity() && depthValue != 0 && depthValue != 65535)
			{
				// coordinates projected to rgb images
				Eigen::Vector3f uv_depth(col, row, 1.0f);
				Eigen::Vector3f uv_color = depthValue / 1000.f * R * uv_depth + T / 1000;   //mapping relationship calculation

				int X = static_cast<int>(uv_color[0] / uv_color[2]);         //X coordinate
				int Y = static_cast<int>(uv_color[1] / uv_color[2]);         //Y coordinate

				if ((X >= 0 && X < 1920) && (Y >= 0 && Y < 1080))
				{

					result.data[i * 3] = bgr.data[3 * (Y * 1920 + X)];
					result.data[i * 3 + 1] = bgr.data[3 * (Y * 1920 + X) + 1];
					result.data[i * 3 + 2] = bgr.data[3 * (Y * 1920 + X) + 2];
				}
			}
			i++;
		}
	}
	//store matched result
	imwrite("matched_image.png", result);

	thread th2 = std::thread([&] {
		while (true)
		{
			namedWindow("Display window3", WINDOW_AUTOSIZE);
			imshow("matched_results", result);
			waitKey(1);
			
		}
		});

	th.join();
	th2.join();
	return 0;
}
