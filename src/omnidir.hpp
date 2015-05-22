#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__
#include "precomp.hpp"
#include <iostream>
namespace cv
{
namespace omnidir
{
    void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec, 
                       InputArray K, InputArray D, double s, double xi,OutputArray jacobian = noArray());

    void undistortPoints(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, double xi, InputArray R, InputArray P, double s);
    
    void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double xi, double s);

    void initUndistortRectifyMap(InputArray K, InputArray D, double xi, double s, InputArray R, InputArray P, const cv::Size& size, int mltype, OutputArray map1, OutputArray map2);
    
	void undistortImage(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, double xi, double s, InputArray Knew, const Size& new_size);
} // omnidir

} //cv
#endif
