/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "opencv2/ccalib/omnidir.hpp"

class omnidirTest:public ::testing::Test{
protected:
    const static cv::Size imageSize;
    const static cv::Matx33d K;
    const static cv::Vec4d D;
    const static cv::Vec3d om;
    const static cv::Vec3d T;
    const static double xi;
    std::string datasets_repository_path;

    virtual void SetUp() {
        datasets_repository_path = combine(cvtest::TS::ptr()->get_data_path(), "cv/cameracalibration/omnidir");
    }
protected:
    std::string combine(const std::string& _item1, const std::string& _item2);
};
TEST_F(omnidirTest, projectPoints)
{
    double cols = this->imageSize.width,
        rows = this->imageSize.height;
    double xi = this->xi;

    const int N = 20;
    cv::Mat distorted0(1, N*N, CV_64FC2), undist1, undist2, distorted2;
    undist2.create(distorted0.size(), CV_MAKETYPE(distorted0.depth(), 3));
    cv::Vec2d* pts = distorted0.ptr<cv::Vec2d>();

    cv::Vec2d c(this->K(0, 2), this->K(1, 2));

    for(int y = 0, k = 0; y < N; ++y)
    {
        for(int x = 0; x < N; ++x)
        {
            cv::Vec2d point(x*cols/(N-1.f), y*rows/(N-1.f));
            pts[k++] = (point - c) * 0.85 + c;
        }
    }
    cv::omnidir::undistortPoints(distorted0, undist1, this->K, this->D, xi, cv::noArray());
    cv::Vec2d* u1 = undist1.ptr<cv::Vec2d>();
    cv::Vec3d* u2 = undist2.ptr<cv::Vec3d>();

    // transform to unit sphere
    for(int i = 0; i  < (int)distorted0.total(); ++i)
    {
        cv::Vec3d temp1 = cv::Vec3d(u1[i][0], u1[i][1], 1.0);
        double r2 = temp1[0]*temp1[0] + temp1[1]*temp1[1];
        double a = (r2 + 1);
        double b = 2*xi*r2;
        double cc = r2*xi*xi-1;
        double Zs = (-b + sqrt(b*b - 4*a*cc))/(2*a);
        u2[i] = cv::Vec3d(temp1[0]*(Zs+xi), temp1[1]*(Zs+xi), Zs);
    }
    cv::omnidir::projectPoints(undist2, distorted2, cv::Vec3d::all(0), cv::Vec3d::all(0), this->K, xi, this->D, cv::noArray());

    EXPECT_LT(cv::norm(distorted0-distorted2), 1e-9);
}
TEST_F(omnidirTest, jacobian)
{
    int n = 10;
    cv::Mat X(1, n, CV_64FC3);
    cv::Mat om(3, 1, CV_64F), T(3, 1, CV_64F);
    cv::Mat f(2, 1, CV_64F), c(2, 1, CV_64F);
    cv::Mat D(4, 1, CV_64F);
    double xi;
    double s;
    cv::RNG r;

    r.fill(X, cv::RNG::NORMAL, 2, 1);
    X = cv::abs(X) * 10;

    r.fill(om, cv::RNG::NORMAL, 0, 1);
    om = cv::abs(om);

    r.fill(T, cv::RNG::NORMAL, 0, 1);
    T = cv::abs(T); T.at<double>(2) = 4; T *= 10;

    r.fill(f, cv::RNG::NORMAL, 0, 1);
    f = cv::abs(f) * 1000;

    r.fill(c, cv::RNG::NORMAL, 0, 1);
    c = cv::abs(c) * 1000;

    r.fill(D, cv::RNG::NORMAL, 0, 1);
    D*= 0.5;

    xi = std::abs(r.gaussian(1));
    s = 0.001 * r.gaussian(1);

    cv::Mat x1, x2, xpred;
    cv::Matx33d K(f.at<double>(0), s, c.at<double>(0),
        0,       f.at<double>(1), c.at<double>(1),
        0,                 0,           1);

    cv::Mat jacobians;
    cv::omnidir::projectPoints(X, x1, om, T, K, xi, D, jacobians);

    // Test on T:
    cv::Mat dT(3, 1, CV_64FC1);
    r.fill(dT, cv::RNG::NORMAL, 0, 1);
    dT *= 1e-9*cv::norm(T);
    cv::Mat T2 = T + dT;
    cv::omnidir::projectPoints(X, x2, om, T2, K, xi, D, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(3,6) * dT).reshape(2,1);
    EXPECT_LT(cv::norm(x2 - xpred), 1e-10);

    // Test on om
    cv::Mat dom(3, 1, CV_64FC1);
    r.fill(dom, cv::RNG::NORMAL, 0, 1);
    dom *= 1e-9*cv::norm(om);
    cv::Mat om2 = om + dom;
    cv::omnidir::projectPoints(X, x2, om2, T, K, xi, D, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(0,3) * dom).reshape(2,1);
    EXPECT_LT(cv::norm(x2 - xpred) , 1e-10);

    // Test on f
    cv::Mat df(2, 1, CV_64FC1);
    r.fill(df, cv::RNG::NORMAL, 0, 1);
    df *= 1e-9 * cv::norm(f);
    cv::Matx33d K2 = K + cv::Matx33d(df.at<double>(0), 0, 0, 0, df.at<double>(1), 0, 0, 0, 1);
    cv::omnidir::projectPoints(X, x2, om, T, K2, xi, D, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(6,8)* df).reshape(2, 1);
    EXPECT_LT(cv::norm(x2 - xpred), 1e-10);

    // Test on s
    double ds = r.gaussian(1);
    ds *= 1e-9 * std::abs(s);
    double s2 = s + ds;
    K2 = K;
    K2(0,1) = s2;
    cv::omnidir::projectPoints(X, x2, om, T, K2, xi, D, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(8,9)*ds).reshape(2, 1);
    EXPECT_LT(cv::norm(x2 - xpred), 1e-10);

    // Test on c
    cv::Mat dc(2, 1, CV_64FC1);
    r.fill(dc, cv::RNG::NORMAL, 0, 1);
    dc *= 1e-9 * cv::norm(c);
    K2 = K + cv::Matx33d(0, 0, dc.at<double>(0), 0, 0, dc.at<double>(1), 0, 0, 1);
    cv::omnidir::projectPoints(X, x2, om, T, K2, xi, D, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(9,11)*dc).reshape(2, 1);
    EXPECT_LT(cv::norm(x2 - xpred), 1e-10);

    // Test on xi
    double dxi = r.gaussian(1);
    dxi *= 1e-9 * std::abs(xi);
    double xi2 = xi + dxi;
    cv::omnidir::projectPoints(X, x2, om, T, K, xi2, D, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(11,12)*dxi).reshape(2, 1);
    EXPECT_LT(cv::norm(x2 - xpred), 1e-10);

    // Test on kp
    cv::Mat dD(4, 1, CV_64FC1);
    r.fill(dD, cv::RNG::NORMAL, 0, 1);
    dD *= 1e-9 * cv::norm(D);
    cv::Mat D2 = D + dD;
    cv::omnidir::projectPoints(X, x2, om, T, K, xi, D2, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(12,16)*dD).reshape(2, 1);
    EXPECT_LT(cv::norm(x2 - xpred), 1e-10);
}

//TEST_F(omnidirTest, calibration)
//{
//    // load pattern points and image points, you should assign your path of the corner file.
//    cv::FileStorage fs("omni_calib_data.xml", cv::FileStorage::READ);
//    std::vector<cv::Mat> v_objectPoints, v_imagePoints;
//    cv::Size imgSize;
//    fs["objectPoints"] >> v_objectPoints;
//    fs["imagePoints"] >> v_imagePoints;
//    fs["imageSize"] >> imgSize;
//
//    cv::Mat K, D;
//    std::vector<cv::Vec3d> omAll,tAll;
//    cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-8);
//    cv::Mat xi ;
//    int flag = 0;
//
//    double rms = cv::omnidir::calibrate(v_objectPoints, v_imagePoints, imgSize, K, xi, D, omAll, tAll, flag, critia);
//
//    EXPECT_LT(rms, 2);
//}

TEST_F(omnidirTest, calibrateStereo)
{
	cv::FileStorage fs1("omni_stereocalib_data_fisheye2.xml", cv::FileStorage::READ);
	std::vector<cv::Mat> v_object, v_image1, v_image2;
	cv::Size imageSize;
	fs1["imagePoints1"] >> v_image1;
	fs1["imagePoints2"] >> v_image2;
	fs1["objectPoints"] >> v_object;
	fs1["imageSize"] >> imageSize;

	cv::Mat K1, K2, D1, D2, xi1, xi2;
	cv::Vec3d om, T;
	std::vector<cv::Vec3d> omL, tL;
	cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-4);
	int flags = cv::omnidir::CALIB_FIX_SKEW;
	double rms = cv::omnidir::stereoCalibrate(v_object, v_image1, v_image2, imageSize, imageSize, K1, xi1, D1, K2, xi2, D2, om, T, omL, tL, flags, critia);

	cv::FileStorage fsw("omnidir_stereo_result2.xml", cv::FileStorage::WRITE);
	fsw << "K1" << K1 << "D1" << D1 << "xi1" << xi1 << "K2" << K2 << "D2" << D2 << "xi2"
		<< xi2 << "om" << om << "T" << T;
	fsw.release();
	EXPECT_LT(rms, 2);
}

//TEST_F(omnidirTest, stereoRectification)
//{
//	cv::FileStorage fs("omnidir_stereo_result.xml", cv::FileStorage::READ);
//	cv::Mat K1, K2, D1, D2;
//	cv::Mat xi1, xi2,R;
//	cv::Vec3d om, T;
//	fs["K1"] >> K1;
//	fs["D1"] >> D1;
//	fs["xi1"] >> xi1;
//	fs["K2"] >> K2;
//	fs["D2"] >> D2;
//	fs["xi2"] >> xi2;
//	fs["om"] >> om;
//	fs["T"] >> T;
//	cv::Rodrigues(om, R);
//	cv::Mat R1, R2;
//	double _xi1, _xi2;
//	_xi1 = xi1.at<double>(0);
//	_xi2 = xi2.at<double>(0);
//	cv::omnidir::stereoRectify(R, T, R1, R2);
//
//	cv::Matx33d KNew(1280.0/3.14, 0, 0, 0, 800/3.14, 0, 0,0,1);
//
//	cv::Size newSize(1280,800);
//
//	cv::Mat imgDis1 = cv::imread("stereo_pair_000_l.jpg",cv::IMREAD_GRAYSCALE);
//	cv::Mat imgDis2 = cv::imread("stereo_pair_000_r.jpg",cv::IMREAD_GRAYSCALE);
//	cv::Mat unDis1, unDis2;
//	cv::omnidir::undistortImage(imgDis1, unDis1, K1, D1, _xi1, cv::omnidir::RECTIFY_LONGLATI, KNew, newSize, R1);
//	cv::omnidir::undistortImage(imgDis2, unDis2, K2, D2, _xi2, cv::omnidir::RECTIFY_LONGLATI, KNew, newSize, R2);
//	cv::imwrite("stereo_pair_000_l_rec.jpg",unDis1);
//	cv::imwrite("stereo_pair_000_r_rec.jpg", unDis2);
//}
//
//TEST_F(omnidirTest, stereoRecons)
//{
//    cv::FileStorage fs("omnidir_stereo_result.xml", cv::FileStorage::READ);
//    cv::Mat K1, K2, D1, D2;
//    cv::Mat xi1, xi2;
//    cv::Vec3d om, T;
//    fs["K1"] >> K1;
//    fs["D1"] >> D1;
//    fs["xi1"] >> xi1;
//    fs["K2"] >> K2;
//    fs["D2"] >> D2;
//    fs["xi2"] >> xi2;
//    fs["om"] >> om;
//    fs["T"] >> T;
//    double _xi1, _xi2;
//    _xi1 = xi1.at<double>(0);
//    _xi2 = xi2.at<double>(0);
//    cv::Mat img1, img2, R;
//    Rodrigues(om, R);
//    img1 = cv::imread("stereo_pair_028_l.jpg", cv::IMREAD_GRAYSCALE);
//    img2 = cv::imread("stereo_pair_028_r.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Size imgSize = img1.size();
//    int numDisparities = 16*7;
//    int SADWindowSize = 3;
//    cv::Mat disparity;
//    int flag = cv::omnidir::RECTIFY_LONGLATI;
//	cv::Mat imageRec1, imageRec2;
//    // the range of theta is (0, pi) and the range of phi is (0, pi)
//    cv::Matx33d KNew(imgSize.width / 2.5, 0, -150, 0, imgSize.height / 2.5, -100, 0, 0, 1);
//
//    cv::omnidir::stereoReconstruct(img1, img2, K1, D1, _xi1, K2, D2, _xi2, R, T, flag,
//        numDisparities, SADWindowSize, disparity, imageRec1, imageRec2, imgSize, KNew, cv::noArray());
//
//    cv::imwrite("disparityMap.png", disparity);
//}

const cv::Size omnidirTest::imageSize(1280, 800);

const cv::Matx33d omnidirTest::K(408.9113574019831, 0, 630.2804046001243,
                                 0, 410.4878218607378, 431.9173219995887,
                                 0,               0,                1);

const cv::Vec4d omnidirTest::D(-0.008287863437777166, 0.01177892568193207, 0.02282460697780795, -0.004184916118489851);

const cv::Vec3d omnidirTest::om(0.0001, -0.02, 0.02);

const cv::Vec3d omnidirTest::T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);

const double omnidirTest::xi = 1.05343;

std::string omnidirTest::combine(const std::string& _item1, const std::string& _item2)
{
    std::string item1 = _item1, item2 = _item2;
    std::replace(item1.begin(), item1.end(), '\\', '/');
    std::replace(item2.begin(), item2.end(), '\\', '/');

    if (item1.empty())
        return item2;

    if (item2.empty())
        return item1;

    char last = item1[item1.size()-1];
    return item1 + (last != '/' ? "/" : "") + item2;
}


int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    cv::waitKey();
}
