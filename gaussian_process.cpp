/*
 Copyright (c) 2014, Washington University in St. Louis
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of the Washington University in St. Louis nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL WASHINGTON UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "gaussian_process.h"

namespace lasp{
	
	template<class T>
	int gaussian_process(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& Xcand, LaspMatrix<T>& mean, LaspMatrix<T>& sig, LaspMatrix<T>& ellIn, optimize_options options){
		T noise = static_cast<T>(options.noise);
		T scale = static_cast<T>(options.scale);
		
		//Check our hyperparameter input.
		//If logHyp is true, then ellIn is assumed to be a vector of hyperparameters with the form:
		//ellIn = [ log(noise) log(scale) log(ell) ]', where ell is a vector of characteristic lengths
		LaspMatrix<T> hyp, ell;
		if (options.logHyp) {
			ellIn.exp(hyp);
			noise = hyp(0);
			scale = hyp(1);
			ell = hyp(0, 2, 0, hyp.rows());
		} else {
			ell = ellIn;
		}
		
		bool gpu = options.gpu;
		
		//Setup kernel parameters
		kernel_opt kernelOptions;
		kernelOptions.kernel = ARD;
		kernelOptions.gamma = options.tau;
		kernelOptions.scale = scale;
		
		//Calculate the norms for kernel computation
		LaspMatrix<T> XinNorm;
		LaspMatrix<T> XcandNorm;
		
		Xin.colSqSum(XinNorm);
		Xcand.colSqSum(XcandNorm);
		
		//Compute our various kernels (covariance matricies)
		LaspMatrix<T> Kaa, Kab, Kba, Kbb;
		Kaa.getKernel(kernelOptions, Xin, XinNorm, Xin, XinNorm, ell, false, false, gpu);
		Kab.getKernel(kernelOptions, Xin, XinNorm, Xcand, XcandNorm, ell, false, false, gpu);
		Kba.getKernel(kernelOptions, Xcand, XcandNorm, Xin, XinNorm, ell, false, false, gpu);
		Kbb.getKernel(kernelOptions, Xcand, XcandNorm, Xcand, XcandNorm, ell, false, false, gpu);
		
		Yin.transpose();
		LaspMatrix<T> q, r, cov;
		
		//Add in our noise parameter
		Kaa.diagAdd(noise*noise);
		//Stanford notes include this, GPML formula excludes it
		//Kbb.diagAdd(noise*noise);
		
		//Find distribution parameters for candidate points
		Kaa.solve(Yin, q);
		Kba.multiply(q, mean);
		
		Kaa.solve(Kab, q);
		Kba.multiply(q, r);
		Kbb.subtract(r, cov);
		sig = cov.diag();
		sig.eWiseOp(sig, 0, 1, 0.5);
		
		Yin.transpose();
		mean.transpose();
		sig.transpose();
		
		return 0;
	}
	
	//Non-ARD GP
	template<class T>
	int gaussian_process(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& Xcand, LaspMatrix<T>& mean, LaspMatrix<T>& sig, optimize_options options){
		LaspMatrix<T> temp;
		return gaussian_process(Xin, Yin, Xcand, mean, sig, temp, options);
	}
	
	//Computes the (log) marginal likelihood of our target variables and the partial derivaties w/ respect to the
	// hyper-parameters of the GP. Hyp = [ noise scale ell ], where tau is the characteristic length scale (I think...)
	template<class T>
	T gaussian_process_likelihood(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& hypIn, LaspMatrix<T>& der, optimize_options options){
		//Check if we're given log(hyp) or not
		LaspMatrix<T> hyp;
		if (options.logHyp) {
			hypIn.exp(hyp);
		} else {
			hyp = hypIn;
		}
		
		T noise = hyp(0);
		T scale = hyp(1);
		
		LaspMatrix<T> ell;
		if(hyp.rows() >= Xin.rows() + 2){
			ell = hyp(0, 2, 1, Xin.rows() + 2);
		}
		
		int n = Yin.size();
		T pi = 3.1415926535897;
		bool gpu = options.gpu;
		
		//Setup kernel parameters
		kernel_opt kernelOptions;
		kernelOptions.kernel = ARD;
		kernelOptions.gamma = hyp(2);
		kernelOptions.scale = scale;
		
		//Calculate the norms for kernel computation
		LaspMatrix<T> XinNorm;
		Xin.colSqSum(XinNorm);
		
		//Compute our kernel (covariance matrix)
		LaspMatrix<T> K;
		K.getKernel(kernelOptions, Xin, XinNorm, Xin, XinNorm, ell, false, false, gpu);
		
		//Add in our noise parameter
		K.diagAdd(noise*noise);
		
		LaspMatrix<T> L(n,n,0.0), alpha, A, Q, diagL, transL, logDet, dK, B, C;
		alpha.copy(Yin);
		alpha.transpose();
		
		K.chol(L);
		L.cholSolve(alpha);
		
		//log-likelihood
		T lik = 0;
		
		Yin.multiply(alpha, A);
		lik -= 0.5 * A(0);
		
		diagL = L.diag();
		diagL.log();
		diagL.colSum(logDet);
		lik -= logDet(0);
		
		lik -= (0.5*n) * std::log(2*pi); //noise?
		
		//Make identity matrix to get inverse
		LaspMatrix<T> Kinv(n, n, 0.0);
		for(int i = 0; i < n; ++i){
			Kinv(i,i) = 1.0;
		}
		
		//GPML notes that the full multiplications should be avoided
		//figure this out later
		L.cholSolve(Kinv);
		
		Q = Kinv;
		alpha.multiply(alpha, Q, false, true, 1.0, -1.0);
		
		der.resize(1, hyp.rows());
		
		//noise (Divide/multiply by 2?)
		der(0) = Q.trace() * noise * noise;
		
		//Scale
		LaspMatrix<T> scaleMat, scaleMatTemp;
		
		LaspMatrix<T> Kcpy;
		Kcpy.getKernel(kernelOptions, Xin, XinNorm, Xin, XinNorm, ell, false, false, gpu);
		Q.multiply(Kcpy, scaleMat, true, false);
		
		der(1) = scaleMat.trace();
		
		//Length scales
		kernel_opt distOptions;
		distOptions.kernel = SQDIST;
		
		for(int i = 0; i < ell.rows(); ++i){
			LaspMatrix<T> dist, xi, lengthMat, lengthMatTemp;
			xi = Xin(0, i, Xin.cols(), i+1).copy();
			xi.multiply(1.0 / ell(i));
			dist.getKernel(distOptions, xi, xi);
			dist.eWiseMultM(Kcpy, dist);
			dist.multiply(0.5);
			Q.multiply(dist, lengthMat, true, false);
			der(i+2) = lengthMat.trace();
		}
				
		return lik;
	}
	
	template int gaussian_process(LaspMatrix<double>& Xin, LaspMatrix<double>& Yin, LaspMatrix<double>& Xcand, LaspMatrix<double>& mean, LaspMatrix<double>& sig, LaspMatrix<double>& ellIn, optimize_options options);
	template int gaussian_process(LaspMatrix<float>& Xin, LaspMatrix<float>& Yin, LaspMatrix<float>& Xcand, LaspMatrix<float>& mean, LaspMatrix<float>& sig, LaspMatrix<float>& ellIn, optimize_options options);
	
	template int gaussian_process(LaspMatrix<double>& Xin, LaspMatrix<double>& Yin, LaspMatrix<double>& Xcand, LaspMatrix<double>& mean, LaspMatrix<double>& sig,  optimize_options options);
	template int gaussian_process(LaspMatrix<float>& Xin, LaspMatrix<float>& Yin, LaspMatrix<float>& Xcand, LaspMatrix<float>& mean, LaspMatrix<float>& sig, optimize_options options);
	
	template double gaussian_process_likelihood(LaspMatrix<double>& Xin, LaspMatrix<double>& Yin, LaspMatrix<double>& hypIn, LaspMatrix<double>& der, optimize_options options);
	template float gaussian_process_likelihood(LaspMatrix<float>& Xin, LaspMatrix<float>& Yin, LaspMatrix<float>& hypIn, LaspMatrix<float>& der, optimize_options options);
	
}
