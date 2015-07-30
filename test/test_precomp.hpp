#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/imgcodecs/imgcodecs.hpp"

namespace cvtest
{
    void Rodrigues(const Mat& src, Mat& dst, Mat* jac=0);
}

#endif
