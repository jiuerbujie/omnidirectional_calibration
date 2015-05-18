#include"opencv2/core/core.hpp"
#include "omnidir.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	Mat objPoint(1, 1, CV_64FC3);
	Vec3d* pts = objPoint.ptr<Vec3d>(0);
	pts[0] = Vec3d(1.0,1.5,2.5);
	//pts[1] = Vec3d(-0.5,0.5,0.5);
	Mat rvec = (Mat_<double>(3,1) << 0.02, -0.01, 0);
	Mat tvec = (Mat_<double>(3,1) << 0.1,0,0);
	Mat K = (Mat_<double>(3,3) << 558.4, 0, 620.4, 0, 560.5, 381.9, 0,0,1);
	Mat D = (Mat_<double>(4,1) << 0.001,0.002,-0.0003,0.0002);
	double xi = 0.8;
	Mat imgPoints,jacobian;
	omnidir::projectPoints(objPoint, imgPoints, rvec, tvec, K, D ,xi, jacobian);
	cout << imgPoints<<endl;
	cout << jacobian <<endl;
	getchar();
}