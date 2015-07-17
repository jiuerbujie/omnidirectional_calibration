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

/**
 * This module was accepted as a GSoC 2015 project for OpenCV, authored by
 * Baisheng Lai, mentored by Bo Li.

 * This module implements points extraction from a "random" pattern image.
 * It returns 3D object points and 2D image pixels that can be used for calibration.
 * Compared with traditional calibration object like chessboard, it is useful when pattern is 
 * partly occluded or only a part of pattern can be observed in multiple cameras calibration.
 * For detail, refer to this paper
 * B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
 * Calibration Toolbox Using A Feature Descriptor-Based Calibration
 * Pattern", in IROS 2013.
 */
#include "precomp.hpp"
#include "randomPatten.hpp"
#include <iostream>
using namespace cv;
using namespace std;
randomPatternCornerFinder::randomPatternCornerFinder(double patternWidth, double patternHeight, int nfeatures,
    int nminiMatch, int depth)
{
    _patternHeight = patternHeight;
    _patternWidth = patternWidth;
    _nminiMatch = nminiMatch;
    _nfeatures = nfeatures;
    _objectPonits.resize(0);
    _imagePoints.resize(0);
    _depth = depth;
}

void randomPatternCornerFinder::computeObjectImagePoints(std::vector<cv::Mat> inputImages, cv::Mat patternImage)
{
    CV_Assert(patternImage.type() == CV_8UC1);

    _patternImageSize = patternImage.size();

    int nImag = (int)inputImages.size();
    Ptr<ORB> detector = ORB::create(_nfeatures);
    //Ptr<BRISK> detector = BRISK::create(100);

    // key points detection and description for pattern
    Mat descriptorPattern;
    std::vector<cv::KeyPoint> keypointsPattern;
    Mat keypointsPatternLocation;
    
    detector->detect(patternImage, keypointsPattern);
    detector->compute(patternImage, keypointsPattern, descriptorPattern);

    // only CV_32F type is support for match
    descriptorPattern.convertTo(descriptorPattern, CV_32F);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
    for (int i = 0; i < nImag; ++i)
    {
        CV_Assert(inputImages[i].type() == CV_8UC1);

        // key points for image
        std::vector<cv::KeyPoint> keypointsImage;
        Mat keypointsImageLocation;
        Mat descriptorImage;

        //equalizeHist(inputImages[i], inputImages[i]);

        detector->detect(inputImages[i], keypointsImage);
        detector->compute(inputImages[i], keypointsImage, descriptorImage);
        // only CV_32F type is support for match
        descriptorImage.convertTo(descriptorImage, CV_32F);

        // match with pattern
        std::vector<DMatch> matchesImgtoPat; 
        
        //matcher->match(descriptorImage, descriptorPattern, matchesImgtoPat);
        crossCheckMatching(matcher, descriptorImage, descriptorPattern, matchesImgtoPat);
        keyPoints2MatchedLocation(keypointsImage, keypointsPattern, matchesImgtoPat,
            keypointsImageLocation, keypointsPatternLocation);

        Mat img_corr;

        // innerMask is CV_8U type
        Mat innerMask1, innerMask2;

        // draw raw correspondence
        //drawCorrespondence(inputImages[i], keypointsImage, patternImage, keypointsPattern, matchesImgtoPat,
        //    innerMask1, innerMask2);

        // outlier remove
        findFundamentalMat(keypointsImageLocation, keypointsPatternLocation,
            FM_7POINT, 1, 0.995, innerMask1);
        getFilteredLocation(keypointsImageLocation, keypointsPatternLocation, innerMask1);

        findHomography(keypointsImageLocation, keypointsPatternLocation, RANSAC, 3, innerMask2);
        getFilteredLocation(keypointsImageLocation, keypointsPatternLocation, innerMask2);

        // draw filtered correspondence
        //drawCorrespondence(inputImages[i], keypointsImage, patternImage, keypointsPattern, matchesImgtoPat,
        //    innerMask1, innerMask2);
        if(keypointsImageLocation.total() > _nminiMatch)
        {
            getObjectImagePoints(keypointsImageLocation, keypointsPatternLocation, i);
        } 
    }
}

void randomPatternCornerFinder::keyPoints2MatchedLocation(const std::vector<cv::KeyPoint>& imageKeypoints,
    const std::vector<cv::KeyPoint>& patternKeypoints, const std::vector<cv::DMatch> matchces,
    cv::Mat& matchedImagelocation, cv::Mat& matchedPatternLocation)
{
    matchedImagelocation.release();
    matchedPatternLocation.release();
    std::vector<Vec2d> image, pattern;
    for (int i = 0; i < (int)matchces.size(); ++i)
    {
        Point2f imgPt = imageKeypoints[matchces[i].queryIdx].pt;
        Point2f patPt = patternKeypoints[matchces[i].trainIdx].pt;
        image.push_back(Vec2d(imgPt.x, imgPt.y));
        pattern.push_back(Vec2d(patPt.x, patPt.y));
    }
    Mat(image).convertTo(matchedImagelocation, CV_64FC2);
    Mat(pattern).convertTo(matchedPatternLocation, CV_64FC2);
}

void randomPatternCornerFinder::getFilteredLocation(cv::Mat& imageKeypoints, cv::Mat& patternKeypoints, const cv::Mat mask)
{
    Mat tmpKeypoint, tmpPattern;
    imageKeypoints.copyTo(tmpKeypoint);
    patternKeypoints.copyTo(tmpPattern);
    imageKeypoints.release();
    patternKeypoints.release();
    std::vector<cv::Vec2d> vecKeypoint, vecPattern;
    for(int i = 0; i < (int)mask.total(); ++i)
    {
        if (mask.at<uchar>(i) == 1)
        {
            vecKeypoint.push_back(tmpKeypoint.at<Vec2d>(i));
            vecPattern.push_back(tmpPattern.at<Vec2d>(i));
        }
    }
    Mat(vecKeypoint).convertTo(imageKeypoints, CV_64FC2);
    Mat(vecPattern).convertTo(patternKeypoints, CV_64FC2);
}

void randomPatternCornerFinder::getObjectImagePoints(const cv::Mat& imageKeypoints, const cv::Mat& patternKeypoints, int i)
{
    Mat imagePoints_i, objectPoints_i;
    int imagePointsType = CV_MAKETYPE(_depth, 2);
    int objectPointsType = CV_MAKETYPE(_depth, 3);
    imageKeypoints.convertTo(imagePoints_i, imagePointsType);
    _imagePoints.push_back(imagePoints_i);
    if (patternKeypoints.total()!=CV_64FC2)
        patternKeypoints.convertTo(patternKeypoints, CV_64FC2);

    std::vector<Vec3d> objectPoints;

    for (int i = 0; i < (int)patternKeypoints.total(); ++i)
    {
        double x = patternKeypoints.at<Vec2d>(i)[0];
        double y = patternKeypoints.at<Vec2d>(i)[1];
        x = x / _patternImageSize.width * _patternWidth;
        y = y / _patternImageSize.height * _patternHeight;
        objectPoints.push_back(Vec3d(x, y, 0));
    }

    Mat(objectPoints).convertTo(objectPoints_i, objectPointsType);
    _objectPonits.push_back(objectPoints_i);
}

void randomPatternCornerFinder::crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
    const Mat& descriptors1, const Mat& descriptors2,
    std::vector<DMatch>& filteredMatches12, int knn )
{
    filteredMatches12.clear();
    std::vector<std::vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

void randomPatternCornerFinder::drawCorrespondence(const Mat& image1, const std::vector<cv::KeyPoint> keypoint1,
    const Mat& image2, const std::vector<cv::KeyPoint> keypoint2, const std::vector<cv::DMatch> matchces,
    const Mat& mask1, const Mat& mask2)
{
    Mat img_corr;
    if(mask1.empty())
    {
        drawMatches(image1, keypoint1, image2, keypoint2, matchces, img_corr);
    }
    else if(!mask1.empty() && mask2.empty())
    {
        std::vector<cv::DMatch> matchesFilter;
        for (int i = 0; i < (int)mask1.total(); ++i)
        {
            if (mask1.at<uchar>(i) == 1)
            {
                matchesFilter.push_back(matchces[i]);
            }
        }
        drawMatches(image1, keypoint1, image2, keypoint2, matchesFilter, img_corr);
    }
    else if(!mask1.empty() && !mask2.empty())
    {
        std::vector<cv::DMatch> matchesFilter;
        int j = 0;
        for (int i = 0; i < (int)mask1.total(); ++i)
        {
            if (mask1.at<uchar>(i) == 1)
            {
                if (mask2.at<uchar>(j) == 1)
                {
                    matchesFilter.push_back(matchces[i]);
                }
                j++;
            }
        }
        drawMatches(image1, keypoint1, image2, keypoint2, matchesFilter, img_corr);
    }
    imshow("correspondence", img_corr);
    waitKey(0);
}

std::vector<cv::Mat> randomPatternCornerFinder::getObjectPoints()
{
    return _objectPonits;
}

std::vector<cv::Mat> randomPatternCornerFinder::getImagePoints()
{
    return _imagePoints;
}