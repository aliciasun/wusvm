//
//  logistic_model.h
//  SP_SVM
//
//  Created by Gabriel Hope on 9/26/14.
//
//

#ifndef __SP_SVM__logistic_model__
#define __SP_SVM__logistic_model__

#include <iostream>
#include "base_model.h"
#include "lasp_func.h"
#include "optimize.h"


namespace lasp {
	
	//Logistic regression
	template<class T>
	class Logistic: public Model<T, int> {
		LaspMatrix<T> Xin, Yin, w;
		int features, n, weights;
		bool bias;
		
		opt& options(){
			return this->options_;
		}
		
		int train_internal();
		int predict_internal(LaspMatrix<T> X, LaspMatrix<T>& output, bool dist);
		T likelihood_internal(LaspMatrix<T>& grad, LaspMatrix<T>& hess, bool calc_grad, bool calc_hess);
		
	public:
		Logistic();
		Logistic(opt opt_in);
		
		LaspMatrix<T> get_hyp();
		vector<string> get_hyp_labels();
		int set_hyp(LaspMatrix<T> hyp);
		
		int train(LaspMatrix<T> X, LaspMatrix<int> y);
		int train(LaspMatrix<T> X, LaspMatrix<int> y, bool optimize);
		int retrain(LaspMatrix<T> X, LaspMatrix<int> y);
		
		int predict(LaspMatrix<T> X, LaspMatrix<int>& output);
		int confidence(LaspMatrix<T> X, LaspMatrix<T>& output);
		int predict_confidence(LaspMatrix<T> X, LaspMatrix<int>& output_predictions, LaspMatrix<T>& output_confidence);
		int distance(LaspMatrix<T> X, LaspMatrix<T>& output);
		
		T score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual);
		T loss(LaspMatrix<T> y_prob, LaspMatrix<int> y_actual);
		T loss(LaspMatrix<int> y_pred, LaspMatrix<T> y_prob, LaspMatrix<int> y_actual);
		
		T test(LaspMatrix<T> X, LaspMatrix<T> y);
		T test_loss(LaspMatrix<T> X, LaspMatrix<T> y);
		
		T likelihood();
		T likelihood(LaspMatrix<T>& grad);
		T likelihood(LaspMatrix<T>& grad, LaspMatrix<T>& hess);
		
		T operator()(LaspMatrix<T> w_in, LaspMatrix<T>& grad);
		T operator()(LaspMatrix<T> w_in, LaspMatrix<T>& grad, LaspMatrix<T>& hess);
	};
	
	
	template<class T>
	Logistic<T>::Logistic() {
		options().kernel = LINEAR;
	}
	
	template<class T>
	Logistic<T>::Logistic(opt opt_in) {
		options() = opt_in;
	}
	
	template <class T>
	LaspMatrix<T> Logistic<T>::get_hyp(){
		LaspMatrix<T> hyp(1, 1);
		hyp(0) = static_cast<T>(options().C);
		return hyp;
	}
	
	template <class T>
	vector<string> Logistic<T>::get_hyp_labels(){
		vector<string> output;
		output.push_back("C");
		return output;
	}
	
	template <class T>
	int Logistic<T>::set_hyp(LaspMatrix<T> hyp){
		options().C = static_cast<double>(hyp(0));
		return 0;
	}
	
	template <class T>
	int Logistic<T>::train(LaspMatrix<T> X, LaspMatrix<int> y){
		weights = Xin.rows();
        LaspMatrix<T> temp = y.convert<T>();
		if (options().usebias && options().bias != 0) {
			Xin = LaspMatrix<T>::hcat(Xin, LaspMatrix<T>::vcat(X, LaspMatrix<T>::ones(X.cols(), 1)));
			bias = true;
		} else {
			Xin = LaspMatrix<T>::hcat(Xin, X);
			bias = false;
		}
		
		Yin = LaspMatrix<T>::hcat(Yin, temp);
		
		features = Xin.rows();
		n = Xin.cols();
		
		return train_internal();
	}
	
	template <class T>
	int Logistic<T>::retrain(LaspMatrix<T> X, LaspMatrix<int> y){
		Xin = LaspMatrix<T>();
		Yin = LaspMatrix<T>();
		return train(X, y);
	}
	
	template <class T>
	int Logistic<T>::predict(LaspMatrix<T> X, LaspMatrix<int>& output){
		LaspMatrix<T> temp;
		int ret_val = predict_internal(X, temp, false);
        output = temp.template convert<int>();
		return ret_val;
	}
	
	template <class T>
	int Logistic<T>::confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
		return -1;
	}
	
	template <class T>
	int Logistic<T>::predict_confidence(LaspMatrix<T> X, LaspMatrix<int>& output_predictions , LaspMatrix<T>& output_confidence){
		return -1;
	}
	
	template <class T>
	int Logistic<T>::distance(LaspMatrix<T> X, LaspMatrix<T>& output){
		return predict_internal(X, output, true);
	}
	
	template <class T>
	T Logistic<T>::score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual){
		return -1;
	}
	
	template <class T>
	T Logistic<T>::loss(LaspMatrix<T> y_prob, LaspMatrix<int> y_actual){
		return -1;
	}
	
	template <class T>
	T Logistic<T>::loss(LaspMatrix<int> y_pred, LaspMatrix<T> y_prob, LaspMatrix<int> y_actual){
		return -1;
	}
	
	template <class T>
	T Logistic<T>::test(LaspMatrix<T> X, LaspMatrix<T> y){
		return -1;
	}
	
	template <class T>
	T Logistic<T>::test_loss(LaspMatrix<T> X, LaspMatrix<T> y){
		return -1;
	}
	
	template<class T>
	T Logistic<T>::likelihood(){
		LaspMatrix<T> g, h;
		return likelihood_internal(g, h, false, false);
	}
	
	template<class T>
	T Logistic<T>::likelihood(LaspMatrix<T>& grad){
		LaspMatrix<T> h;
		return likelihood_internal(grad, h, true, false);
	}
	
	template<class T>
	T Logistic<T>::likelihood(LaspMatrix<T>& grad, LaspMatrix<T>& hess){
		return likelihood_internal(grad, hess, true, true);
	}
	
	template<class T>
	T Logistic<T>::operator()(LaspMatrix<T> w_in, LaspMatrix<T>& grad){
		w = w_in;
		return likelihood(grad);
	}
	
	template<class T>
	T Logistic<T>::operator()(LaspMatrix<T> w_in, LaspMatrix<T>& grad, LaspMatrix<T>& hess){
		w = w_in;
		return likelihood(grad, hess);
	}
	
	template<class T>
	T Logistic<T>::likelihood_internal(LaspMatrix<T>& grad, LaspMatrix<T>& hess, bool calc_grad, bool calc_hess){
		T lambda = static_cast<T>(1.0 / options().C);
		
		LaspMatrix<T> hx = 1 / (exp(-(t(w) * Xin)) + 1);
		LaspMatrix<T> lik = -sum(mul(Yin, log(hx)) + mul((1 - Yin), log(1 - hx))) / n;
		LaspMatrix<T> reg = (lambda / (2 * n)) * t(rsel(w, 0, weights)) * rsel(w, 0, weights);
		T result = lik(0) + reg(0);
		
		if (calc_grad) {
			LaspMatrix<T> bias_grad;
			if (bias){
				bias_grad = (sum(hx - Yin) * rsel(Xin, weights)) / n;
			}
			
			LaspMatrix<T> g = rsel(Xin, 0, weights);
			LaspMatrix<T> mult = hx - Yin;
			g.rowWiseMult(mult);
            g = lasp::vcat((LaspMatrix<T>)((rsum(g) / n) + ((lambda / n) + rsel(w, 0, weights))), bias_grad);
			grad.operator=(g);
		}
		
		if (calc_hess) {
			
		}
		
		return result;
	}
	
	template<class T>
	int Logistic<T>::train_internal(){
		bool gpu = options().usegpu;
		
		//Check that we have a CUDA device
		if (gpu && DeviceContext::instance()->getNumDevices() < 1) {
			if (options().verb > 0) {
				cerr << "No CUDA device found, reverting to CPU-only version" << endl;
			}
			
			options().usegpu = false;
			gpu = false;
		} else if(!gpu){
			DeviceContext::instance()->setNumDevices(0);
		} else if (options().maxGPUs > -1){
			DeviceContext::instance()->setNumDevices(options().maxGPUs);
		}

		//Timing Variables
		clock_t baseTime = clock();
		
		//kernel options struct for computing the kernel
		kernel_opt kernelOptions = options().kernel_options();
		
		//Training examples
		LaspMatrix<T> x = Xin;
		
		//Training labels
		LaspMatrix<T> y = Yin;
		
		//Move data to the gpu
		if (gpu) {
			int err = x.transferToDevice();
			err += y.transferToDevice();
			
			if (err != MATRIX_SUCCESS) {
				x.transferToHost();
				y.transferToHost();
				
				if (options().verb > 0) {
					cerr << "Device memory insufficient for data, reverting to CPU-only computation" << endl;
				}
				
				gpu = false;
			}
		}
		
		//Norm of each training vector
		LaspMatrix<T> xNorm (n,1, 0.0);
		x.colSqSum(xNorm);
		
		w = LaspMatrix<T> (1, features, 0.0);
		
		//Move support vector stuff to gpu
		if (gpu) {
			w.transferToDevice();
		}
						
		conjugate_optimize(*this, w, optimize_options());
		
		if (gpu) {
			w.transferToHost();
		}
		
		if (options().verb > 0){
			cout << "Training Complete" << endl;
		}
		
		return 0;
	}
	
	template<class T>
	int Logistic<T>::predict_internal(LaspMatrix<T> X, LaspMatrix<T>& output, bool dist){
		output.resize(X.cols(), 1);
		
		//Set GPU parameters
		if (options().usegpu && DeviceContext::instance()->getNumDevices() < 1) {
			if (options().verb > 0) {
				cerr << "No CUDA device found, reverting to CPU-only version" << endl;
			}
			
			options().usegpu = false;
		} else if (options().maxGPUs > -1){
			DeviceContext::instance()->setNumDevices(options().maxGPUs);
		}
		
		//Create data matricies
		LaspMatrix<T> dXS = w;
		LaspMatrix<T> dXe = X;
		
		//Account for potential difference in number of realized features in train/test data
		int totalFeatures = 0;
		totalFeatures = std::max(dXS.rows(), dXe.rows() + 1);
		if (totalFeatures > dXS.rows()) {
			LaspMatrix<T> lastRowS = dXS(0, dXS.rows()-1, dXS.cols(), dXS.rows()).copy();
			dXS.resize(dXS.cols(), dXS.rows()-1);
			dXS.resize(dXS.cols(), totalFeatures, true, true, 0.0);
			dXS.setRow(dXS.rows()-1, lastRowS);
		}
		
		dXe.resize(dXe.cols(), totalFeatures, true, true, 0.0);
		LaspMatrix<T> lastRow = dXe(0, dXe.rows()-1, dXe.cols(), dXe.rows());
		lastRow.add(1.0);
		
		LaspMatrix<T> dXe_copy, dXS_copy;
		dXe_copy = dXe;
		dXS_copy = dXS;
		
		
		//Calculate norms
		LaspMatrix<T> dXeNorm;
		dXe_copy.colSqSum(dXeNorm);
		
		LaspMatrix<T> dXSNorm;
		dXS_copy.colSqSum(dXSNorm);
		
		//Set kernel options
		kernel_opt kernelOptions = options().kernel_options();
		
		//Calculate chunks to break up test points into if we're using the small kernel option
		int maxElem = 0;
		int numChunks = 0;
		int chunkStart = 0;
		
		//Make sure our kernel allocation succeeds
		LaspMatrix<T> Ke;
		while(true){
			maxElem = options().set_size;
			numChunks = options().smallKernel ? 1 + ((X.cols() - 1) / maxElem) : 1;
			
			//Kernel matrix
			try {
				Ke.resize(X.cols() / numChunks, w.cols());
				break;
			} catch (std::bad_alloc) {
				options().smallKernel = true;
				options().set_size /= 2;
			}
		}
		
		if (options().usegpu) {
			Ke.transferToDevice();
		}
		
		//Loop through the chunks
		for (int chunk = 0; chunk < numChunks; ++chunk) {
			int chunkSize = (chunk == numChunks - 1) ? X.cols() - chunkStart : X.cols() / numChunks;
			int chunkEnd = chunkStart + chunkSize;
			
			if (chunkSize == 0) {
				continue;
			}
			
			//Get chunk kernel matrix
			LaspMatrix<T> dXe_chunk = dXe_copy(chunkStart, 0, chunkEnd, dXe.rows());
			LaspMatrix<T> dXeNorm_chunk = dXeNorm(chunkStart, 0, chunkEnd, 1);
			Ke.getKernel(kernelOptions, dXS_copy, dXSNorm, dXe_chunk, dXeNorm_chunk, false, false, options().usegpu);
			LaspMatrix<T> classes = Ke;
			classes.transferToHost();  //Avoid race condition in operator()
			
			//Set classes
			if(!dist){
#pragma omp parallel for
				for (int i=0; i < chunkSize; ++i) {
					output(i + chunkStart) = classes(i, 0) > 0.5 ? 1 : -1;
				}
			}
			else {
#pragma omp parallel for
				for (int i=0; i < chunkSize; ++i) {
					output(i + chunkStart) = classes(i, 0);
				}
			}
			
			chunkStart = chunkEnd;
		}
		
		return 0;
	}
}




#endif /* defined(__SP_SVM__logistic_model__) */
