Omnidirectional camera calibration toolbox
===========================
This package consists of three parts.

- omnidirectional camera calibration
- random pattern calibration object
- multiple camera calibration

Requires

- OpenCV 3.0

This package is selected as a project in GSoC 2015 of OpenCV, mentored by [Bo Li](https://github.com/prclibo). This repository is a standalone project during my development, you may notice that the directory structure is the same as OpenCV. Now, it is merged into  OpenCV's [opencv_contrib repository](https://github.com/opencv/opencv_contrib). You can use it by compiling OpenCV with opencv_contrib or adding source files and include files in this repository to your project.

Usage
-----------------
Add files in ```src``` and ```include``` to where they should be, it depends on your development environment.

For omnidirectional camera calibration, use function ```cv::omnidir::calibrate``` in ```omnidir.cpp```. This is API is fully compatible with OpenCV's ```cv::calibrateCamera```.

For random pattern calibration object, use class ```RandomPatternCornerFinder``` in ```randpattern.cpp```.

For multiple camera calibration, use class ```MultiCameraCalibration``` in ```multicalib.cpp```. So far, multiple camera calibration only support random pattern object. The name of calibration images are required to be "cameraIdx-timestamp.*" and cameraIdx starts from 0. **Other name rules will not work!**

Samples
---------------
In the ```samples``` directory, there are several samples that describe how to use these APIs. These samples are something like samples in OpenCV, i.e., they are some programs that can be directly used for your application like calibration. Data to be used to run samples is in ```data``` directory.

Tutorial
---------------
In the ```tutorial``` directory, step by step tutorials are given.

A Video
-------------------
This is a [video](https://www.youtube.com/watch?v=E7YvGX6_RHE) that shows some results of this package.

Future Work
--------------
Revise the API of multiple camera calibration to be more general so that object points and image points from each camera and view are given.
