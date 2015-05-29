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

#include "omnidir.hpp"
namespace cv { namespace
{
    struct JacobianRow
    {
        Matx13d dom,dT;
        Matx12d df;
        double ds;
        Matx12d dc;
        double dxi;
        Matx14d dkp;    // distortion k1,k2,p1,p2
    };
}}

/////////////////////////////////////////////////////////////////////////////
//////// projectPoints
void cv::omnidir::projectPoints(InputArray objectPoints, OutputArray imagePoints, 
                InputArray _rvec, InputArray _tvec, InputArray _K, InputArray _D, double xi, OutputArray jacobian)
{
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    // each row is an image point
    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));
    size_t n = objectPoints.total();
    Vec3d om = _rvec.depth() == CV_32F ? (Vec3d)*_rvec.getMat().ptr<Vec3f>() : *_rvec.getMat().ptr<Vec3d>();   
    Vec3d T  = _tvec.depth() == CV_32F ? (Vec3d)*_tvec.getMat().ptr<Vec3f>() : *_tvec.getMat().ptr<Vec3d>(); 
    Matx33d K = _K.getMat();
    Vec<double, 4> kp= (Vec<double,4>)*_D.getMat().ptr<Vec<double,4> >();

    Vec2d f,c;
    f = Vec2d(K(0,0),K(1,1));
    c = Vec2d(K(0,2),K(1,2));
    double s = K(0,1);
    const Vec3d* Xw_all = objectPoints.getMat().ptr<Vec3d>();
    Vec2d* xpd = imagePoints.getMat().ptr<Vec2d>();

    Matx33d R;
    Matx<double, 3, 9> dRdom;
    Rodrigues(om, R, dRdom);
    // use opencv248
    // use opencv300beta
    //Affine3d aff(om, T);
    
    JacobianRow *Jn = 0;
    if (jacobian.needed())
    {
        int nvars = 2+2+1+4+3+3+1; // f,c,s,kp,om,T,xi
        jacobian.create(2*int(n), nvars, CV_64F);
        Jn = jacobian.getMat().ptr<JacobianRow>(0);
    }

    double k1=kp[0],k2=kp[1];
    double p1 = kp[2], p2 = kp[3];

    for (size_t i = 0; i < n; i++)
    {
        // convert to camera coordinate
        Vec3d Xw = (Vec3d)Xw_all[i];
        // use opencv248
        Vec3d Xc = (Vec3d)(R*Xw + T);
        // use opencv300beta
        //Vec3d Xc = aff*Xw;

        // convert to unit sphere
        Vec3d Xs = Xc/cv::norm(Xc);

        // convert to normalized image plane
        Vec2d xu = Vec2d(Xs[0]/(Xs[2]+xi), Xs[1]/(Xs[2]+xi));

        // add distortion
        Vec2d xd;
        double r2 = xu[0]*xu[0]+xu[1]*xu[1];
        double r4 = r2*r2;
        double r6 = r2*r4;
        xd[0] = xu[0]*(1+k1*r2+k2*r4) + 2*p1*xu[0]*xu[1] + p2*(r2+2*xu[0]*xu[0]);
        xd[1] = xu[1]*(1+k1*r2+k2*r4) + p1*(r2+2*xu[1]*xu[1]) + 2*p2*xu[0]*xu[1];

        // convert to pixel coordinate
        xpd[i][0] = f[0]*xd[0]+s*xd[1]+c[0];
        xpd[i][1] = f[1]*xd[1]+c[1];

        if (jacobian.needed())
        {
            double dXcdR_a[] = {Xw[0],Xw[1],Xw[2],0,0,0,0,0,0,
                                0,0,0,Xw[0],Xw[1],Xw[2],0,0,0,
                                0,0,0,0,0,0,Xw[0],Xw[1],Xw[2]};
            Matx<double,3, 9> dXcdR(dXcdR_a);
            Matx33d dXcdom = dXcdR * dRdom.t();
            double r_1 = 1.0/norm(Xc);
            double r_3 = pow(r_1,3);
            Matx33d dXsdXc(r_1-Xc[0]*Xc[0]*r_3, -(Xc[0]*Xc[1])*r_3, -(Xc[0]*Xc[2])*r_3,
                           -(Xc[0]*Xc[1])*r_3, r_1-Xc[1]*Xc[1]*r_3, -(Xc[1]*Xc[2])*r_3,
                           -(Xc[0]*Xc[2])*r_3, -(Xc[1]*Xc[2])*r_3, r_1-Xc[2]*Xc[2]*r_3);
            Matx23d dxudXs(1/(Xs[2]+xi),    0,    -Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
                           0,    1/(Xs[2]+xi),    -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));
            // pre-compute some reusable things
            double temp1 = 2*k1*xu[0] + 4*k2*xu[0]*r2;
            double temp2 = 2*k1*xu[1] + 4*k2*xu[1]*r2;
            Matx22d dxddxu(k2*r4+6*p2*xu[0]+2*p1*xu[1]+xu[0]*temp1+k1*r2+1,    2*p1*xu[0]+2*p2*xu[1]+xu[0]*temp2,
                           2*p1*xu[0]+2*p2*xu[1]+xu[1]*temp1,    k2*r4+2*p2*xu[0]+6*p1*xu[1]+xu[1]*temp2+k1*r2+1);
            Matx22d dxpddxd(f[0], s,
                            0, f[1]);
            Matx23d dxpddXc = dxpddxd * dxddxu * dxudXs * dXsdXc;

            // derivative of xpd respect to om
            Matx23d dxpddom = dxpddXc * dXcdom;
            Matx33d dXcdT(1.0,0.0,0.0,
                          0.0,1.0,0.0,
                          0.0,0.0,1.0);
            // derivative of xpd respect to T

            Matx23d dxpddT = dxpddXc * dXcdT;
            Matx21d dxudxi(-Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
                           -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));

            // derivative of xpd respect to xi
            Matx21d dxpddxi = dxpddxd * dxddxu * dxudxi;
            Matx<double,2,4> dxddkp(xu[0]*r2, xu[0]*r4, 2*xu[0]*xu[1], r2+2*xu[0]*xu[0],
                                    xu[1]*r2, xu[1]*r4, r2+2*xu[1]*xu[1], 2*xu[0]*xu[1]);
            
            // derivative of xpd respect to kp
            Matx<double,2,4> dxpddkp = dxpddxd * dxddkp;
            
            // derivative of xpd respect to f
            Matx22d dxpddf(xd[0], 0,
                           0, xd[1]);
            
            // derivative of xpd respect to c
            Matx22d dxpddc(1, 0,
                           0, 1);
            
            Jn[0].dom = dxpddom.row(0);
            Jn[1].dom = dxpddom.row(1);
            Jn[0].dT = dxpddT.row(0);
            Jn[1].dT = dxpddT.row(1);
            Jn[0].dkp = dxpddkp.row(0);
            Jn[1].dkp = dxpddkp.row(1);
            Jn[0].dxi = dxpddxi(0,0);
            Jn[1].dxi = dxpddxi(1,0);
            Jn[0].df = dxpddf.row(0);
            Jn[1].df = dxpddf.row(1);
            Jn[0].dc = dxpddc.row(0);
            Jn[1].dc = dxpddc.row(1);
            Jn[0].ds = xd[1];
            Jn[1].ds = 0;
            Jn += 2;
         }
    }
}

/////////////////////////////////////////////////////////////////////////////
//////// distortPoints
void cv::omnidir::distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double xi)
{
    CV_Assert(undistorted.type() == CV_64FC2);
    distorted.create(undistorted.size(), undistorted.type());
    size_t n = undistorted.total();

    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_64F && D.total() == 4);

    Vec4d kp = (Vec4d)*D.getMat().ptr<Vec4d>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);

    Vec2d f, c;
    Matx33d camMat = K.getMat();
    f = Vec2d(camMat(0,0), camMat(1,1));
    c = Vec2d(camMat(0,2), camMat(1,2));
    double s = camMat(0,1);
    const Vec2d *srcd = undistorted.getMat().ptr<Vec2d>();
    Vec2d *desd = distorted.getMat().ptr<Vec2d>();

    for (size_t i = 0; i < n; i++)
    {
        Vec2d pi = srcd[i];    // image points
        Vec2d xu((pi[0]*f[1]-c[0]*f[1]-s*(pi[1]-c[1]))/(f[0]*f[1]), (pi[1]-c[1])/f[1]); // unified image plane
        double r2 = xu[0]*xu[0] + xu[1]*xu[1];
        double r4 = r2*r2;
        // add distortion
        Vec2d xd;
        xd[0] = (1+k[0]*r2+k[1]*r4)*xu[0] + 2*p[0]*xu[0]*xu[1] + p[1]*(r2+2*xu[0]*xu[0]);
        xd[1] = (1+k[0]*r2+k[1]*r4)*xu[1] + p[0]*(r2+2*xu[1]*xu[1]) + 2*p[1]*xu[0]*xu[1];
        // project to image
        Vec3d pr = camMat * Vec3d(xd[0], xd[1], 1);
        desd[i] = Vec2d(pr[0],pr[1]);
    }
}



/////////////////////////////////////////////////////////////////////////////
//////// undistortPoints
void cv::omnidir::undistortPoints( InputArray distorted, OutputArray undistorted, 
                         InputArray K, InputArray D, double xi, InputArray R, InputArray P)
{
    CV_Assert(distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());
    
    CV_Assert(P.empty() || P.size() == Size(3,3) || P.size() == Size(4,3));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && K.depth() == CV_64F);

    cv::Vec2d f, c;
    Matx33d camMat = K.getMat();
    f = cv::Vec2d(camMat(0,0), camMat(1,1));
    c = cv::Vec2d(camMat(0,2), camMat(1,2));
    double s = camMat(0,1);
    Vec4d kp = (Vec4d)*D.getMat().ptr<Vec4d>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);

    cv::Matx33d RR = cv::Matx33d::eye();
    // R is om
    if(!R.empty() && R.total()*R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        //RR = cv::Affine3d(rvec).rotation();
        cv::Rodrigues(rvec, RR);
    }
    else if (!R.empty() && R.size() == Size(3,3))
    {
        R.getMat().convertTo(RR, CV_64F);
    }
    cv::Matx33d PP;
    if(!P.empty())
    {
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
        //RR = PP * RR;
    }
    else
    {
        //RR = camMat * RR; 
        PP = camMat;    //use old camera matrix
    }
    const cv::Vec2d *srcd = distorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2d *dstd = undistorted.getMat().ptr<cv::Vec2d>();

    size_t n = distorted.total();
    for (size_t i = 0; i < n; i++)
    {
        Vec2d pi = (Vec2d)srcd[i];    // image point
        //Vec2d pw((pi[0]-c[0]/f[0], (pi[1]-c[1])/f[1]));    // unified image plane
        Vec2d pp((pi[0]*f[1]-c[0]*f[1]-s*(pi[1]-c[1]))/(f[0]*f[1]), (pi[1]-c[1])/f[1]);
        Vec2d pu = pp;    // points without distortion
        // remove distortion iteratively
        for (int j = 0; j < 10; j++)
        {
            double r2 = pu[0]*pu[0] + pu[1]*pu[1];
            double r4 = r2*r2;
            pu[0] = (pp[0] - 2*p[0]*pu[0]*pu[1] - p[1]*(r2+2*pu[0]*pu[0])) / (1 + k[0]*r2 + k[1]*r4);
            pu[1] = (pp[1] - 2*p[1]*pu[0]*pu[1] - p[0]*(r2+2*pu[1]*pu[1])) / (1 + k[0]*r2 + k[1]*r4);
        }
        // project to unit sphere
        double r2 = pu[0]*pu[0] + pu[1]*pu[1];
        double a = (r2 + 1);
        double b = 2*xi*r2;
        double cc = r2*xi*xi-1;
        double Zs = (-b + sqrt(b*b - 4*a*cc))/(2*a);
        Vec3d Xw = Vec3d(pu[0]*(Zs + xi), pu[1]*(Zs +xi), Zs);
        // rotate
        Xw = RR * Xw;
        // project back to sphere
        Vec3d Xs = Xw / cv::norm(Xw);
        // reproject to image
        Vec3d ppu = Vec3d(Xs[0]/(Xs[2]+xi), Xs[1]/(Xs[2]+xi), 1.0);
        Vec3d pr = PP * ppu;    // apply rotation and may new camera matrix
        //Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);
        //dstd[i] = Vec2d(pr[0]/pr[2], pr[1]/pr[2]);
        dstd[i] = Vec2d(pr[0], pr[1]);
    }
}


/////////////////////////////////////////////////////////////////////////////
//////// cv::omnidir::initUndistortRectifyMap
void cv::omnidir::initUndistortRectifyMap(InputArray K, InputArray D, double xi, InputArray R, InputArray P, 
    const cv::Size& size, int m1type, OutputArray map1, OutputArray map2)
{
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32F || m1type <=0 );
    map1.create( size, m1type <= 0 ? CV_16SC2 : m1type );
    map2.create( size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F );
    
    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert((P.depth() == CV_32F || P.depth() == CV_64F) && (R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(K.size() == Size(3, 3) && (D.empty() || D.total() == 4));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));  
    
    cv::Vec2d f, c;
    double s;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
        s = camMat(0,1);
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
        s = camMat(0,1);
    }

    Vec4d kp = Vec4d::all(0);
    if (!D.empty())
        kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);
    cv::Matx33d RR  = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        cv::Rodrigues(rvec, RR);
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    cv::Matx33d PP = cv::Matx33d::eye();
    if (!P.empty())
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
    else
        PP = K.getMat();
    
    cv::Matx33d iR = (PP*RR).inv(cv::DECOMP_SVD);
    //cv::Matx33d iK = PP.inv(cv::DECOMP_SVD);
    // so far it is undistorted to perspective image
    for (int i = 0; i < size.height; ++i)
    {
        float* m1f = map1.getMat().ptr<float>(i);
        float* m2f = map2.getMat().ptr<float>(i);
        short*  m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;

        double _x = i*iR(0, 1) + iR(0, 2),
               _y = i*iR(1, 1) + iR(1, 2),
               _w = i*iR(2, 1) + iR(2, 2);
        for(int j = 0; j < size.width; ++j, _x+=iR(0,0), _y+=iR(1,0), _w+=iR(2,0))
        {
            // project back to unit sphere
            double r = sqrt(_x*_x + _y*_y + _w*_w);
            double Xs = _x / r;
            double Ys = _y / r;
            double Zs = _w / r;
            // project to image plane
            double xu = Xs / (Zs + xi),
                   yu = Ys / (Zs + xi);
            // add distortion
            double r2 = xu*xu + yu*yu;
            double r4 = r2*r2;
            double xd = (1+k[0]*r2+k[1]*r4)*xu + 2*p[0]*xu*yu + p[1]*(r2+2*xu*xu);
            double yd = (1+k[0]*r2+k[1]*r4)*yu + p[0]*(r2+2*yu*yu) + 2*p[1]*xu*yu;
            // to image pixel
            double u = f[0]*xd + s*yd + c[0];
            double v = f[1]*yd + c[1];
            
            if( m1type == CV_16SC2 )
            {
                int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
            }
            else if( m1type == CV_32FC1 )
            {
                m1f[j] = (float)u;
                m2f[j] = (float)v;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::omnidir::undistortImage

void cv::omnidir::undistortImage(InputArray distorted, OutputArray undistorted,
    InputArray K, InputArray D, double xi, InputArray Knew, const Size& new_size)
{
    Size size = new_size.area() != 0 ? new_size : distorted.size();

    cv::Mat map1, map2;
    omnidir::initUndistortRectifyMap(K, D, xi, cv::Matx33d::eye(), Knew, size, CV_16SC2, map1, map2 );
    cv::remap(distorted, undistorted, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::omnidir::internal::initializeCalibration

void cv::omnidir::internal::initializeCalibration(InputArrayOfArrays patternPoints, InputArrayOfArrays imagePoints, Size size, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray K)
{
    
    double u0 = size.width / 2;
    double v0 = size.height / 2;

    int n_img = (int)imagePoints.total();

    std::vector<cv::Vec3d> v_omAll(n_img), v_tAll(n_img);
    
    std::vector<double> gammaAll(n_img);

    K.create(3, 3, CV_64F);
    Mat _K;
    for (int image_idx = 0; image_idx < n_img; ++image_idx)
    {
        cv::Mat objPoints = patternPoints.getMat(image_idx);
        cv::Mat imgPoints = imagePoints.getMat(image_idx);
        // objectPoints should be 3-channel data, imagePoints should be 2-channel data
        CV_Assert(objPoints.type() == CV_64FC3 && imgPoints.type() == CV_64FC2);

        vector<cv::Mat> xy, uv;
        cv::split(objPoints, xy);
        cv::split(imgPoints, uv);

        int n_point = imgPoints.rows * imgPoints.cols;
        cv::Mat x = xy[0].reshape(1, n_point), y = xy[1].reshape(1, n_point),
                u = uv[0].reshape(1, n_point) - u0, v = uv[1].reshape(1, n_point) - v0;

        cv::Mat sqrRho = u.mul(u) + v.mul(v);
        
        // compute extrinsic parameters
        // Form equation for te essential matrix, see Scaramuzza's paper:
        // A Toolbox for Easily Calibrating Omnidirectional Cameras 
        cv::Mat M(n_point, 6, CV_64F);
        Mat(-v.mul(x)).copyTo(M.col(0));
        Mat(-v.mul(y)).copyTo(M.col(1));
        Mat(u.mul(x)).copyTo(M.col(2));
        Mat(u.mul(y)).copyTo(M.col(3));
        Mat(-v).copyTo(M.col(4));
        Mat(u).copyTo(M.col(5));

        Mat W,U,V;
        cv::SVD::compute(M, W, U, V,SVD::FULL_UV);
        V = V.t();

        double miniReprojectError = 1e5;
        // the signs of r1, r2, r3 are unkown, so they can be flipped.
        for (int coef = 1; coef >= -1; coef-=2)
        {
            double r11 = V.at<double>(0, 5) * coef;
            double r12 = V.at<double>(1, 5) * coef;
            double r21 = V.at<double>(2, 5) * coef;
            double r22 = V.at<double>(3, 5) * coef;
            double t1 = V.at<double>(4, 5) * coef;
            double t2 = V.at<double>(5, 5) * coef;

            Mat roots;
            double r31s;
            solvePoly(Matx13d(-(r11*r12+r21*r22)*(r11*r12+r21*r22), r11*r11+r21*r21-r12*r12-r22*r22, 1), roots);

            if (roots.at<Vec2d>(0)[0] > 0)
                r31s = sqrt(roots.at<Vec2d>(0)[0]);
            else
                r31s = sqrt(roots.at<Vec2d>(1)[0]);
            
            for (int coef2 = 1; coef2 >= -1; coef2-=2)
            {
                double r31 = r31s * coef2;
                double r32 = -(r11*r12 + r21*r22) / r31;

                cv::Vec3d r1(r11, r21, r31);
                cv::Vec3d r2(r12, r22, r32);
                cv::Vec3d t(t1, t2, 0);
                double scale = 1 / cv::norm(r1);
                r1 = r1 * scale;
                r2 = r2 * scale;
                t = t * scale;

                // compute intrisic parameters
                // Form equations in Scaramuzza's paper
                // A Toolbox for Easily Calibrating Omnidirectional Cameras 
                Mat A(n_point*2, 3, CV_64F);
                Mat((r1[1]*x + r2[1]*y + t[1])/2).copyTo(A.rowRange(0, n_point).col(0));
                Mat((r1[0]*x + r2[0]*y + t[0])/2).copyTo(A.rowRange(n_point, 2*n_point).col(0));
                Mat(-A.col(0).rowRange(0, n_point).mul(sqrRho)).copyTo(A.col(1).rowRange(0, n_point));
                Mat(-A.col(0).rowRange(n_point, 2*n_point).mul(sqrRho)).copyTo(A.col(1).rowRange(n_point, 2*n_point));
                Mat(-v).copyTo(A.rowRange(0, n_point).col(2));
                Mat(-u).copyTo(A.rowRange(n_point, 2*n_point).col(2));
                
                // Operation to avoid bad numerical-condition of A
                Vec3d maxA, minA;
                for (int j = 0; j < A.cols; j++)
                {
                    cv::minMaxLoc(cv::abs(A.col(j)), &minA[j], &maxA[j]);
                    A.col(j) = A.col(j) / maxA[j];
                }

                Mat B(n_point*2 , 1, CV_64F);
                Mat(v.mul(r1[2]*x + r2[2]*y)).copyTo(B.rowRange(0, n_point));
                Mat(u.mul(r1[2]*x + r2[2]*y)).copyTo(B.rowRange(n_point, 2*n_point));

                Mat res = A.inv(DECOMP_SVD) * B;
                res = res.mul(1/Mat(maxA));

                double gamma = sqrt(res.at<double>(0) / res.at<double>(1));
                t[2] = res.at<double>(2);

                cv::Vec3d r3 = r1.cross(r2);

                Matx33d R(r1[0], r2[0], r3[0],
                          r1[1], r2[1], r3[1],
                          r1[2], r2[2], r3[2]);
                Vec3d om;
                Rodrigues(R, om);
                // pattern points after applied rotation and translation, if it is behind of camera, then
                // the signs of parameters are invalid. 
                Mat patternPointsExt = Mat(R) * objPoints.reshape(1, 3) + cv::repeat(Mat(t), 1, n_point);
                for (int i = 0; i < n_point; ++i)
                {
                    if (patternPointsExt.at<double>(2,i) <0)
                        continue;
                }

                // project pattern points to images
                Mat projedImgPoints;
                Matx33d Kc(gamma, 0, u0, 0, gamma, v0, 0, 0, 1);
                Matx14d D(0, 0, 0, 0);
                cv::omnidir::projectPoints(objPoints, projedImgPoints, om, t, Kc, D, 1, cv::noArray());
                double reprojectError = cv::norm(projedImgPoints - imgPoints, cv::NORM_L2);

                // if this reproject error is smaller
                if (reprojectError < miniReprojectError)
                {
                    miniReprojectError = reprojectError;
                    v_omAll[image_idx] = om;
                    v_tAll[image_idx] = t;
                    gammaAll[image_idx] = gamma;
                }
            }
        }
    }
        double gammaFinal = 0;
        for (int i = 0; i < (int)gammaAll.size(); i++)
            gammaFinal += gammaAll[i];
        gammaFinal = gammaFinal / gammaAll.size();
        _K = Mat(Matx33d(gammaFinal, 0, u0, 0, gammaFinal, v0, 0, 0, 1));
        _K.convertTo(K, CV_64F);
        cv::Mat(v_omAll).convertTo(omAll, CV_64FC3);
        cv::Mat(v_tAll).convertTo(tAll, CV_64FC3);
    
}
