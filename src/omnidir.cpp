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
        Matx14d dkp;	// distortion k1,k2,p1,p2
    };
}}

/////////////////////////////////////////////////////////////////////////////
//////// projectPoints
void cv::omnidir::projectPoints(InputArray objectPoints, OutputArray imagePoints, 
                InputArray _rvec, InputArray _tvec, InputArray _K, InputArray _D, double xi, double s, OutputArray jacobian)
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
            Matx23d dxudXs(1/(Xs[2]+xi),	0,	-Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
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
//////// undistortPoints
void cv::omnidir::undistortPoints( InputArray distorted, OutputArray undistorted, 
                         InputArray K, InputArray D, InputArray R, InputArray P)
{
    CV_Assert(distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());
    
    CV_Assert(P.empty() || P.size() == Size(3,3) || P.size() == Size(4,3));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && K.depth() == CV_64F);

    cv::Vec2d f, c;
    Matx33d camMat = K.getMat();
    f = cv::Vec2d(camMat(0,0), camMat(1,1));

    Vec4d k = (Vec4d)*D.getMat().ptr<Vec4d>();

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

    if(!P.empty())
    {
		cv::Matx33d PP;
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }
    else
    {
        RR = camMat * RR; 
    }
    const cv::Vec2d *srcd = distorted.getMat().ptr<cv::Vec2d>();
	cv::Vec2d *dstd = undistorted.getMat().ptr<cv::Vec2d>();

    size_t n = distorted.total();

    for (size_t i = 0; i < n; i++)
    {
        Vec2d pi = (Vec2d)srcd[i];	// image point
        Vec2d pw((pi[0]-c[0]/f[0], (pi[1]-c[1])/f[1]));	// world point
        Vec2d pu = pw;	// points withour distortion
        Vec2d pu_temp;
        // remove distortion iteratively
        for (int j = 0; j < 10; j++)
        {
            double r2 = pu[0]*pu[0] + pu[1]*pu[1];
            double r4 = r2*r2;
            pu_temp[0] = (pu[0] - 2*k[2]*pu[0]*pu[1] - k[3]*(r2+2*pu[0]*pu[0])) / (1 + k[0]*r2 + k[1]*r4);
            pu_temp[1] = (pu[1] - 2*k[3]*pu[0]*pu[1] - k[2]*(r2+2*pu[1]*pu[1])) / (1 + k[0]*r2 + k[1]*r4);
            pu = pu_temp;
        }
        // reproject to image
        Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0);	// apply rotation and may 
        //Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);
		dstd[i] = Vec2d(pr[0]/pr[2], pr[1]/pr[2]);
	}


}
