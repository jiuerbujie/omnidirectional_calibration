#include "../src/omnidir.hpp"
#include "../src/multiCameraCalibration.hpp"
#include "../src/randomPatten.hpp"

int main()
{
    multiCameraCalibration multiCalib(multiCameraCalibration::OMNIDIRECTIONAL, 5, "multi_camera_subset.xml", 800, 600, 10);

    multiCalib.loadImages();
    multiCalib.initialize();
    

}