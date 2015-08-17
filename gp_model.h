//
//  gp_model.h
//  SP_SVM
//
//  Created by Gabriel Hope on 9/26/14.
//
//

#ifndef __SP_SVM__gp_model__
#define __SP_SVM__gp_model__

#include <iostream>
#include "base_model.h"
#include "optimize.h"


namespace lasp {
	
	//Gaussian process regression model
	template<class T>
	class GP: public Model<T, T> {
		LaspMatrix<T> Xin, Yin;
		LaspMatrix<T> ell_;
		LaspMatrix<T> sig;

		int optimize_likelihood();
		LaspMatrix<T> get_kernel_deriv(int deriv, LaspMatrix<T> K);
		
		opt& options(){
			return this->options_;
		}
		
	public:
		GP();
		GP(opt opt_in);
		
		LaspMatrix<T> get_hyp();
		vector<string> get_hyp_labels();
		int set_hyp(LaspMatrix<T> hyp);
		
		int train(LaspMatrix<T> X, LaspMatrix<T> y);
		int train(LaspMatrix<T> X, LaspMatrix<T> y, bool optimize);
		int retrain(LaspMatrix<T> X, LaspMatrix<T> y);
		
		int predict(LaspMatrix<T> X, LaspMatrix<T>& output);
		int confidence(LaspMatrix<T> X, LaspMatrix<T>& output);
		int predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions, LaspMatrix<T>& output_confidence);
		int distance(LaspMatrix<T> X, LaspMatrix<T>& output);
		
		T score(LaspMatrix<T> y_pred, LaspMatrix<T> y_actual);
		T loss(LaspMatrix<T> y_prob, LaspMatrix<T> y_actual);
		T loss(LaspMatrix<T> y_pred, LaspMatrix<T> y_prob, LaspMatrix<T> y_actual);
		
		T test(LaspMatrix<T> X, LaspMatrix<T> y);
		T test_loss(LaspMatrix<T> X, LaspMatrix<T> y);
		
		T likelihood();
		T likelihood(LaspMatrix<T>& der);
		
		int expected_improvement(LaspMatrix<T> X, LaspMatrix<T>& ei_out);
		T best_expected_improvement(LaspMatrix<T> X, LaspMatrix<T>& ei_out);
		
		T operator()(LaspMatrix<T>& hyp, LaspMatrix<T>& grad);
	};
	
	
	template<class T>
	GP<T>::GP() {
		options().kernel = ARD;
	}
	
	template<class T>
	GP<T>::GP(opt opt_in) {
		options() = opt_in;
	}
	
	template <class T>
	LaspMatrix<T> GP<T>::get_hyp(){
		LaspMatrix<T> hyp(1, 2);
		hyp(0) = static_cast<T>(options().noise);
		hyp(1) = static_cast<T>(options().mean);
		
		switch (options().kernel) {
			case ARD:
				hyp.resize(1, 3);
				hyp(2) = static_cast<T>(options().scale);
				hyp = LaspMatrix<T>::vcat(hyp, ell_);
				break;
			case RBF:
				hyp.resize(1, 3);
				hyp(2) = static_cast<T>(options().gamma);
				break;
			case EXP:
				hyp.resize(1, 4);
				hyp(2) = static_cast<T>(options().alpha);
				hyp(3) = static_cast<T>(options().beta);
				break;
			case POLYNOMIAL:
				hyp.resize(1, 3);
				hyp(2) = static_cast<T>(options().coef);
				break;
			case SIGMOID:
				hyp.resize(1, 4);
				hyp(2) = static_cast<T>(options().coef);
				hyp(3) = static_cast<T>(options().gamma);
				break;
			default:
				break;
		}
		
		return hyp;
	}
	
	template <class T>
	vector<string> GP<T>::get_hyp_labels(){
		vector<string> output;
		output.push_back("noise");
		output.push_back("mean");
		
		switch (options().kernel) {
			case ARD:
				output.push_back("scale");
				for (int i = 0; i < ell_.size(); ++i) {
					output.push_back("ell");
				}
				break;
			case RBF:
				output.push_back("gamma");
				break;
			case EXP:
				output.push_back("alpha");
				output.push_back("beta");
				break;
			case POLYNOMIAL:
				output.push_back("coef");
				break;
			case SIGMOID:
				output.push_back("coef");
				output.push_back("gamma");
				break;
			default:
				break;
		}
		
		return output;
	}
	
	template <class T>
	int GP<T>::set_hyp(LaspMatrix<T> hyp){
		options().noise = static_cast<double>(hyp(0));
		options().mean = static_cast<double>(hyp(1));
		
		switch (options().kernel) {
			case ARD:
				options().scale = static_cast<double>(hyp(2));
				ell_ = hyp(0, 3, 1, hyp.rows()).copy();
				break;
			case RBF:
				options().gamma = static_cast<double>(hyp(2));
				break;
			case EXP:
				options().alpha = static_cast<double>(hyp(2));
				options().beta = static_cast<double>(hyp(3));
				break;
			case POLYNOMIAL:
				options().coef = static_cast<double>(hyp(2));
				break;
			case SIGMOID:
				options().coef = static_cast<double>(hyp(2));
				options().gamma = static_cast<double>(hyp(3));
				break;
			default:
				break;
		}
		
		return 0;
	}
	
	template<class T>
	LaspMatrix<T> GP<T>::get_kernel_deriv(int deriv, LaspMatrix<T> K){
		T degree = static_cast<T>(options().degree);
		T alpha = static_cast<T>(options().alpha);
		T beta = static_cast<T>(options().beta);
		
		switch (options().kernel) {
			case ARD:
			{
				if (deriv == 0) {
					return K;
				}
				
				deriv--;
				if (deriv < ell_.rows()) {
					//Length scales
					kernel_opt distOptions;
					distOptions.kernel = SQDIST;
					
					LaspMatrix<T> dist, xi, lengthMat, lengthMatTemp;
					xi = Xin(0, deriv, Xin.cols(), deriv+1).copy();
					xi.multiply(1.0 / ell_(deriv));
					dist.getKernel(distOptions, xi, xi);
					dist.eWiseMultM(K, dist);
					dist.multiply(0.5);
					return dist;
				}
				
				break;
			}
			case RBF:
			{
				if (deriv == 0) {
					//Length scales
					kernel_opt distOptions;
					distOptions.kernel = SQDIST;
					
					LaspMatrix<T> dist, lengthMat, lengthMatTemp;
					dist.getKernel(distOptions, Xin, Xin);
					dist.eWiseMultM(K, dist);
					dist.multiply(0.5);
					return dist;
				}
				
				break;
			}
			case EXP:
			{
				LaspMatrix<T> Kadd;
				Xin.add_outer(Xin, Kadd);
				
				LaspMatrix<T> denom;
				Kadd.add(beta, denom);
				denom.pow(2 * alpha);
				
				if (deriv == 0) {
					LaspMatrix<T> p1, p2, p3, p4;
					
					T bta = std::pow(beta, alpha) * std::log(beta) * alpha;
					Kadd.add(beta, p1);
					p1.pow(alpha);
					p1.multiply(bta);
					
					bta = std::pow(beta, alpha) * alpha;
					Kadd.add(beta, p2);
					p2.pow(alpha, p3);
					p2.log(p4);
//					
					p3.eWiseMultM(p4, p2);
					p2.multiply(bta);
					
					p1.subtract(p2);
					p1.eWiseDivM(denom, p1);
					
					return p1;
				}
				
				if (deriv == 1) {
					LaspMatrix<T> p1, p2, p3, p4;
					
					T bta = std::pow(beta, alpha - 1) * beta * alpha;
					Kadd.add(beta, p1);
					p1.pow(alpha);
					p1.multiply(bta);
					
					bta = std::pow(beta, alpha) * beta * alpha;
					Kadd.add(beta, p2);
					p2.pow(alpha - 1);
					p2.multiply(bta);
					
					p1.subtract(p2);
					p1.eWiseDivM(denom, p1);
					
					return p1;
				}
				
				break;
			}
			case POLYNOMIAL:
			{
				if (deriv == 0) {
					LaspMatrix<T> Kder, dot;
					kernel_opt poly_opt = options().kernel_options();
					poly_opt.degree--;
					Kder.getKernel(poly_opt, Xin, Xin, ell_, false, false, options().usegpu);
					Kder.multiply(degree);
					
					Xin.multiply(Xin, dot, true);
					Kder.eWiseMultM(dot, Kder);
					
					return Kder;
				}
				
				break;
			}
			case SIGMOID:
			{
				cerr << "I don't want to figure out this derivative. Why do you need the sigmoid kernel for your GP anyway?" << endl;
				throw 1;
				break;
			}
			default:
			{
				break;
			}
		}
		
		
		return LaspMatrix<T>();
	}
	
	template <class T>
	int GP<T>::train(LaspMatrix<T> X, LaspMatrix<T> y){
		return train(X, y, true);
	}
	
	template <class T>
	int GP<T>::train(LaspMatrix<T> X, LaspMatrix<T> y, bool optimize){
		if (options().kernel == EXP && X.rows() > 1) {
			return -1;
		}
		
		Xin = LaspMatrix<T>::hcat(Xin, X);
		Yin = LaspMatrix<T>::hcat(Yin, y);
		
		ell_ = LaspMatrix<T>::ones(1, Xin.rows());
		
		if (optimize) {
			//FIXME
			opt new_options;
			new_options.kernel = this->options().kernel;
			this->options() = new_options;
			
			optimize_likelihood();
			//cout << "cov: " << options().alpha << ", " << options().beta << ". Noise: " << options().noise << ", Mean: " << options().mean << endl;
		}
		
		return 0;
	}
	
	template <class T>
	int GP<T>::retrain(LaspMatrix<T> X, LaspMatrix<T> y){
		Xin = LaspMatrix<T>();
		Yin = LaspMatrix<T>();
		return train(X, y);
	}
	
	template <class T>
	int GP<T>::predict(LaspMatrix<T> X, LaspMatrix<T>& output){
		LaspMatrix<T> temp;
		return predict_confidence(X, output, temp);
	}
	
	template <class T>
	int GP<T>::confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
		LaspMatrix<T> temp;
		return predict_confidence(X, temp, output);
	}
	
	template <class T>
	int GP<T>::predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions , LaspMatrix<T>& output_confidence){
		
		LaspMatrix<T>& mean = output_predictions;
		LaspMatrix<T>& sig = output_confidence;
		
		LaspMatrix<T> Y;
		Yin.subtract(options().mean, Y);
		Y.transpose();
		
		bool gpu = options().usegpu;
		T noise = options().noise;
		
		//Setup kernel parameters
		kernel_opt kernelOptions = options().kernel_options();
		
		//Calculate the norms for kernel computation
		LaspMatrix<T> XinNorm;
		LaspMatrix<T> XNorm;
		
		Xin.colSqSum(XinNorm);
		X.colSqSum(XNorm);
		
		//Compute our various kernels (covariance matricies)
		LaspMatrix<T> Kaa, Kab, Kba, Kbb;
		Kaa.getKernel(kernelOptions, Xin, XinNorm, Xin, XinNorm, ell_, false, false, gpu);
		Kab.getKernel(kernelOptions, Xin, XinNorm, X, XNorm, ell_, false, false, gpu);
		Kba.getKernel(kernelOptions, X, XNorm, Xin, XinNorm, ell_, false, false, gpu);
		Kbb.getKernel(kernelOptions, X, XNorm, X, XNorm, ell_, false, false, gpu);
		
		LaspMatrix<T> q, r, cov;
		
		//Add in our noise parameter
		Kaa.diagAdd(noise*noise);
		//Stanford notes include this, GPML formula excludes it
		//Kbb.diagAdd(noise*noise);
		
		//Find distribution parameters for candidate points
		Kaa.solve(Y, q);
		Kba.multiply(q, mean);
		mean.add(options().mean);
		
		Kaa.solve(Kab, q);
		Kba.multiply(q, r);
		Kbb.subtract(r, cov);
		sig = cov.diag();
		
		sig.eWiseOp(sig, 0, 1, 0.5);
		
		mean.transpose();
		sig.transpose();
		
		if(sig(0) != sig(0)) {
//			cov.printMatrix("cov");
//			cov.diag().printMatrix("diag");
//			cout << "WTF!!!" << endl;
		}
		
		return 0;
	}
	
	template <class T>
	int GP<T>::distance(LaspMatrix<T> X, LaspMatrix<T>& output){
		return -1;
	}
	
	template <class T>
	T GP<T>::score(LaspMatrix<T> y_pred, LaspMatrix<T> y_actual){
		return -1;
	}
	
	template <class T>
	T GP<T>::loss(LaspMatrix<T> y_prob, LaspMatrix<T> y_actual){
		return -1;
	}
	
	template <class T>
	T GP<T>::loss(LaspMatrix<T> y_pred, LaspMatrix<T> y_prob, LaspMatrix<T> y_actual){
		return -1;
	}
	
	template <class T>
	T GP<T>::test(LaspMatrix<T> X, LaspMatrix<T> y){
		return -1;
	}
	
	template <class T>
	T GP<T>::test_loss(LaspMatrix<T> X, LaspMatrix<T> y){
		return -1;
	}
	
	template <class T>
	T GP<T>::likelihood(){
		return likelihood(LaspMatrix<T>());
	}
	
	template <class T>
	T GP<T>::likelihood(LaspMatrix<T>& der){
		
		T noise = options().noise;
		T scale = options().scale;
		
		LaspMatrix<T> Y;
		Yin.subtract(options().mean, Y);
		
		int n = Y.size();
		T pi = 3.1415926535897;
		bool gpu = options().usegpu;
		
		//Setup kernel parameters
		kernel_opt kernelOptions = options().kernel_options();
		
		//Calculate the norms for kernel computation
		LaspMatrix<T> XinNorm;
		Xin.colSqSum(XinNorm);
		
		//Compute our kernel (covariance matrix)
		LaspMatrix<T> K;
		K.getKernel(kernelOptions, Xin, XinNorm, Xin, XinNorm, ell_, false, false, gpu);
		
		//Add in our noise parameter
		K.diagAdd(noise*noise);
		
		LaspMatrix<T> L(n,n,0.0), alpha, A, Q, diagL, transL, logDet, dK, B, C;
		alpha.copy(Y);
		alpha.transpose();
		
		K.chol(L);
		L.cholSolve(alpha);
		
		//log-likelihood
		T lik = 0;
		
		Y.multiply(alpha, A);
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
		
		der = get_hyp();
		//der.printMatrix("hyp");
		//noise (Divide/multiply by 2?)
		der(0) = Q.trace() * noise * noise;
		//mean
		LaspMatrix<T> neg(alpha.rows(), 1, 1);
		//neg.printMatrix("neg");
		neg.multiply(alpha);
		//alpha.printMatrix("alpha");
		der(1) = neg(0);
		//neg.printMatrix("neg2");
		
		//Kernel derivs
		LaspMatrix<T> Kcpy;
		Kcpy.getKernel(kernelOptions, Xin, XinNorm, Xin, XinNorm, ell_, false, false, gpu);
			
		for (int deriv_ind = 0; ; ++deriv_ind) {
			LaspMatrix<T> kernel_deriv = get_kernel_deriv(deriv_ind, Kcpy);
			
			if (kernel_deriv.size() == 0) {
				break;
			}
			
			LaspMatrix<T> deriv;
			Q.multiply(kernel_deriv, deriv, true, false);
			
			der(deriv_ind + 2) = deriv.trace() / 2; //Unclear if this should actually have "/ 2"
		}
		
		//der.printMatrix("der");
				
		return lik;
	}
	
	template<class T>
	int GP<T>::expected_improvement(LaspMatrix<T> X, LaspMatrix<T> &ei_out){
		//Find the best current input point
		T Ybest = Yin.minElem();
		
		LaspMatrix<T> mean, sig;
		predict_confidence(X, mean, sig);
		
		//Now compute the expected improvement for each candidate
		LaspMatrix<T> Z, Zcdf, Zpdf;
		mean.subtract(Ybest);
		mean.negate();
		mean.eWiseDivM(sig, Z);
		
		Z.normPDF(Zpdf);
		Z.normCDF(Zcdf);
		
		Z.eWiseMultM(Zcdf, Z);
		Z.add(Zpdf);
		sig.eWiseMultM(Z, ei_out);
		
		return 0;
	}
	
	template<class T>
	T GP<T>::best_expected_improvement(LaspMatrix<T> X, LaspMatrix<T> &ei_out){
		LaspMatrix<T> ei;
		this->expected_improvement(X, ei);
		
		int row, best_col;
		T ei_value = ei.maxElem(best_col, row);
		ei_out.copy(X(best_col, 0, best_col+1, X.rows()));
		return ei_value;
	}
	
	
	template<class T>
	int GP<T>::optimize_likelihood(){
		LaspMatrix<T> hyp = get_hyp();
		//cout << "Gradient check: " << check_gradient(*this, hyp, (T)1e-6) << endl;
		LaspMatrix<T> temp;
//		cout << "lik: " << operator()(hyp, temp) << endl;
//		temp.printMatrix("grad");
		conjugate_optimize(*this, hyp);
		set_hyp(hyp);
		return 0;
	}
	
	template<class T>
	T GP<T>::operator()(LaspMatrix<T>& hyp, LaspMatrix<T>& grad) {
		set_hyp(hyp);
		T lik = likelihood(grad);
		grad.negate();
		return -lik;
	}
}




#endif /* defined(__SP_SVM__gp_model__) */
