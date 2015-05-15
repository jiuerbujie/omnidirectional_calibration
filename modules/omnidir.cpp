#include "omnidir.hpp"
namespace cv { namespace
{
    struct JacobianRow
    {
		Matx13d dom,dT;
		Matx12d df,dc;
		double dxi;
		Matx14d dkp;	// distortion k1,k2,p1,p2
    };
}}
void cv::omnidir::projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray _rvec, InputArray _tvec, InputArray _K, InputArray _D, double xi, OutputArray jacobian)
{
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
	// each row is an image point
    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));
    size_t n = objectPoints.total();
    
    Vec3d om = _rvec.depth() == CV_32F ? (Vec3d)*_rvec.getMat().ptr<Vec3f>() : *_rvec.getMat().ptr<Vec3d>();   
    Vec3d T  = _tvec.depth() == CV_32F ? (Vec3d)*_tvec.getMat().ptr<Vec3f>() : *_tvec.getMat().ptr<Vec3d>(); 
    Matx33d K = _K.getMat();
	Vec<double, 4> kp= (Vec<double,4>)*_D.getMat().ptr<Vec<double,4>>();
    Vec2d f,c;
	f = Vec2d(K(0,0),K(1,1));
	c = Vec2d(K(0,2),K(1,2));
	const Vec3d* Xw_all = objectPoints.getMat().ptr<Vec3d>();
	Vec2d* xpd = imagePoints.getMat().ptr<Vec2d>();

	Matx33d R;
	Matx<double, 3, 9> dRdom;
	Rodrigues(om, R, dRdom);
	Affine3d aff(om, T);

	JacobianRow *Jn = 0;
	if (jacobian.needed())
	{
		int nvars = 2+2+4+3+3+1; // f,c,kp,om,T,xi
		jacobian.create(2*int(n), nvars, CV_64F);
		Jn = jacobian.getMat().ptr<JacobianRow>(0);
	}
	double k1=kp[0],k2=kp[1];
	double p1 = kp[2], p2 = kp[3];

	for (size_t i = 0; i < n; i++)
	{
		// convert to camera coordiante
		Vec3d Xw = (Vec3d)Xw_all[i];
		Vec3d Xc = aff*Xw;
		// convert to unit sphere
		Vec3d Xs = Xc/cv::norm(Xc);
		// convert to normalized image plane
		Vec2d xu = Vec2d(Xs[0]/(Xs[2]+xi), Xs[1]/(Xs[2]+xi));
		// add distoration
		Vec2d xd;
		double r2 = xu[0]*xu[0]+xu[1]*xu[1];
		double r4 = r2*r2;
		double r6 = r2*r4;
		xd[0] = xu[0]*(1+k1*r2+k2*r4) + 2*p1*xu[0]*xu[1] + p2*(r2+2*xu[0]*xu[0]);
		xd[1] = xu[1]*(1+k1*r2+k2*r4) + p1*(r2+2*xu[1]*xu[1]) + 2*p2*xu[0]*xu[1];
		// convert to pixel coordinate
		xpd[i][0] = f[0]*xd[0]+c[0];
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
							0,	1/(Xs[2]+xi),	-Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));

			// pre-compute some reusable things
			double temp1 = 2*k1*xu[0] + 4*k2*xu[0]*r2;
			double temp2 = 2*k1*xu[1] + 4*k2*xu[1]*r2;

			Matx22d dxddxu(k2*r4 + 6*p2*xu[0] + 2*p1*xu[1] + xu[0]*temp1+k1*r2+1,	2*p1*xu[0] + 2*p2*xu[1]+xu[0]*temp2,
							2*p1*xu[0] + 2*p2*xu[1] + xu[1]*temp1,	k2*r4 + 2*p2*xu[0] + 6*p1*xu[1] + xu[1]*temp2+1);
			Matx22d dxpddxd(f[0], 0.0,
							0, f[1]);
			Matx23d dxpddXc = dxpddxd * dxddxu * dxudXs * dXsdXc;
			// derivative of xpd respect to om
			Matx23d dxpddom = dxpddXc * dXcdom;
			Matx33d dXcdT(1,0,0,
						0,1,0,
						0,0,1);
			// derivative of xpd respect to T
			Matx23d dxpddT = dxpddXc * dXcdT;
			Matx21d dxudxi(-Xs[0]/(Xs[2]+xi)/(Xs[2]+xi), -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));
			// derivative of xpd respect to xi
			Matx21d dxpddxi = dxpddxd * dxddxu * dxudxi;
			Matx<double,2,4> dxddkp(xu[0]*r2, xu[0]*r4, 2*xu[0]*xu[1], r2+2*xu[0]*xu[0],
									xu[1]*r2,	xu[1]*r4,	r2+2*xu[1]*xu[1],	2*xu[0]*xu[1]);
			
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

			Jn += 2;
		}
	}
}
