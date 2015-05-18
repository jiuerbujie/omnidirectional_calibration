#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
namespace cv
{
namespace omnidir
{
    void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec, 
                       InputArray K, InputArray D, double xi, OutputArray jacobian = noArray());
} // omnidir

} //cv
#endif
