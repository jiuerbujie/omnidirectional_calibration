#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
namespace cv
{
namespace omnidir
{
    void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec, 
                       InputArray K, InputArray D, double xi, double s, OutputArray jacobian = noArray());

    void undistortPoints( InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray R, InputArray P);
} // omnidir

} //cv
#endif
