#include "opencv2/ts.hpp"
#include "../src/omnidir.hpp"

class omnidirTest:public ::testing::Test{
protected:
    const static cv::Size imageSize;
	const static cv::Matx33d K;
	const static cv::Vec4d D;
	const static cv::Matx33d R;
	const static cv::Vec3d T;
	const static double xi;
}

const cv::Size omnidirTest::imageSize(1280, 800);

const cv::Matx33d omnidirTest::K(558.478087865323,               0, 620.458515360843,
                                 0, 560.506767351568, 381.939424848348,
                                 0,               0,                1);

