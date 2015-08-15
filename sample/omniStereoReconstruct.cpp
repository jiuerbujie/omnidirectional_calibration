#include "opencv2/omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    cv::FileStorage fs("fisheye_pair_result.xml", cv::FileStorage::READ);
    cv::Mat K1, K2, D1, D2;
    cv::Mat xi1, xi2;
    cv::Vec3d om, T;
    fs["K1"] >> K1;
    fs["D1"] >> D1;
    fs["xi1"] >> xi1;
    fs["K2"] >> K2;
    fs["D2"] >> D2;
    fs["xi2"] >> xi2;
    fs["om"] >> om;
    fs["T"] >> T;
    cv::Mat img1, img2;
    cv::Mat R;
    Rodrigues(om, R);
    img1 = cv::imread("stereo_pair_028_l.jpg", cv::IMREAD_GRAYSCALE);
    img2 = cv::imread("stereo_pair_028_r.jpg", cv::IMREAD_GRAYSCALE);
    cv::Size imgSize = img1.size();
    int numDisparities = 16*5;
    int SADWindowSize = 5;
    cv::Mat disMap;
    int flag = cv::omnidir::RECTIFY_LONGLATI;
    // the range of theta is (0, pi) and the range of phi is (0, pi)
    cv::Matx33d KNew(imgSize.width / 2.5, 0, -150, 0, imgSize.height / 2.5, -100, 0, 0, 1);

    Mat pointCloud;
	Mat image1Rec, image2Rec;
    //int pointType = omnidir::XYZRGB;
    cv::omnidir::stereoReconstruct(img1, img2, K1, D1, xi1, K2, D2, xi2, R, T, flag,
        numDisparities, SADWindowSize, disMap, image1Rec, image2Rec, imgSize, KNew, pointCloud);
    cv::imwrite("disparity.jpg", disMap);
	cv::imwrite("imgRec1.jpg", image1Rec);
    cv::imwrite("imgRec2.jpg", image2Rec);
	ofstream file("pointsColor.xyz");

    for (int i = 0; i < (int)pointCloud.total(); ++i)
    {
        float x = pointCloud.at<Vec6f>(i)[0];
        float y = pointCloud.at<Vec6f>(i)[1];
        float z = pointCloud.at<Vec6f>(i)[2];
        int r = (int)pointCloud.at<Vec6f>(i)[3];
        int g = (int)pointCloud.at<Vec6f>(i)[4];
        int b = (int)pointCloud.at<Vec6f>(i)[5];

        file << x << " " << y << " " << z << " " << r << " " << g << " " << b << endl;
    }
    file.close();
}