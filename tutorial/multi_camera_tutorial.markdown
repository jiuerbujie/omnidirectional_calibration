Multi-camera Calibration {#tutorial_multi_camera_main}
====================================

This tutorial will show how to use the multiple camera calibration toolbox. This toolbox is based on the usage of "random" pattern calibration object, so the tutorial is mainly two parts: an introduction to "random" pattern and multiple camera calibration.

Random Pattern Calibration Object
-------------------------------
The random pattern is an image that is randomly generated. It is "random" so that it has many feature points. After generating it, one print it out and use it as a calibration object. The following two images are random pattern and a photo taken for it.

![image](img/random_pattern.jpg)
![image](img/pattern_img.jpg)

To generate a random pattern, use the class ```cv::randomPatternGenerator``` in ```ccalib``` module. Run it as
```
randomPatternGenerator generator(width, height);
generator.generatePattern();
pattern = generator.getPattern();
```
Here 
