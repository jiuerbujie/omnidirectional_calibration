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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
#include "../src/omnidir.hpp"

class omnidirTest:public ::testing::Test{
protected:
    const static cv::Size imageSize;
	const static cv::Matx33d K;
	const static cv::Vec4d D;
	const static cv::Vec3d om;
	const static cv::Vec3d T;
	const static double xi;
};

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

    xi = abs(r.gaussian(1));
    s = 0.001 * r.gaussian(1);

    cv::Mat x1, x2, xpred;
    cv::Matx33d K(f.at<double>(0), s, c.at<double>(0),
                       0,       f.at<double>(1), c.at<double>(1),
                       0,                 0,           1);
    
    cv::Mat jacobians;
    cv::omnidir::projectPoints(X, x1, om, T, K, D, xi, s, jacobians);

    // Test on T:
    cv::Mat dT(3, 1, CV_64FC1);
    r.fill(dT, cv::RNG::NORMAL, 0, 1);
    dT *= 1e-9*cv::norm(T);
    cv::Mat T2 = T + dT;
    cv::omnidir::projectPoints(X, x2, om, T2, K, D, xi, s, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(3,6) * dT).reshape(2,1);
	std::cout << cv::norm(x2 - xpred) <<std::endl;
	CV_Assert(cv::norm(x2 - xpred) < 1e-10);

}


const cv::Size omnidirTest::imageSize(1280, 800);

const cv::Matx33d omnidirTest::K(558.478087865323,               0, 620.458515360843,
                                 0, 560.506767351568, 381.939424848348,
                                 0,               0,                1);

const cv::Vec4d omnidirTest::D(-0.0014613319981768, -0.00329861110580401, 0.00605760088590183, -0.00374209380722371);

const cv::Vec3d omnidirTest::om(0.0001, -0.02, 0.02);

const cv::Vec3d omnidirTest::T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);

int main(int argc, char* argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
	cv::waitKey();
}



