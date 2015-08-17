//
//  pegasos_model.h
//  SP_SVM
//
//  Created by Gabriel Hope on 9/26/14.
//
//

#ifndef __SP_SVM__pegasos_model__
#define __SP_SVM__pegasos_model__

#include <iostream>
#include "base_model.h"
#include "optimize.h"
#include <set>

namespace lasp {
	
	//Stochastic gradient descent model
	template<class T>
	class SGD: public Model<T, int> {
		LaspMatrix<T> Xin, Yin, xS, xnormS, alphas, alphas_inds;
		vector<int> S;
		T bias;
		int features, n, iter, next_ind, total_data;

		opt& options(){
			return this->options_;
		}
		
		int train_internal();
		int predict_internal(LaspMatrix<T> X, LaspMatrix<T>& output, bool dist);
		
	public:
		SGD();
		SGD(opt opt_in);
		
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
		
		T test(LaspMatrix<T> X, LaspMatrix<int> y);
		T test_loss(LaspMatrix<T> X, LaspMatrix<int> y);

		vector<int> get_sv();
	};
	
	
	template<class T>
	SGD<T>::SGD() {
		options().kernel = LINEAR;
		iter = 0;
		next_ind = 0;
		total_data = 0;
		bias = 0;
	}
	
	template<class T>
	SGD<T>::SGD(opt opt_in) {
		options() = opt_in;
		iter = 0;
		next_ind = 0;
		total_data = 0;
		bias = 0;
	}
	
	template <class T>
	LaspMatrix<T> SGD<T>::get_hyp(){
		LaspMatrix<T> hyp(1, 2);
		hyp(0) = static_cast<T>(options().C);
		hyp(1) = static_cast<T>(options().eta);
		hyp(2) = static_cast<T>(options().set_size);
		
		switch (options().kernel) {
			case RBF:
				hyp.resize(1, 4);
				hyp(3) = static_cast<T>(options().gamma);
				break;
			case POLYNOMIAL:
				hyp.resize(1, 5);
				hyp(3) = static_cast<T>(options().coef);
				hyp(4) = static_cast<T>(options().degree);
				break;
			case SIGMOID:
				hyp.resize(1, 5);
				hyp(3) = static_cast<T>(options().coef);
				hyp(4) = static_cast<T>(options().gamma);
				break;
			default:
				break;
		}
		
		return hyp;
	}
	
	template <class T>
	vector<string> SGD<T>::get_hyp_labels(){
		vector<string> output;
		output.push_back("C");
		output.push_back("eta");
		output.push_back("set_size");
		
		switch (options().kernel) {
			case RBF:
				output.push_back("gamma");
				break;
			case POLYNOMIAL:
				output.push_back("coef");
				output.push_back("degree");
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
	int SGD<T>::set_hyp(LaspMatrix<T> hyp){
		options().C = static_cast<double>(hyp(0));
		options().eta = static_cast<double>(hyp(1));
		options().set_size = static_cast<int>(hyp(2));
		
		switch (options().kernel) {
			case RBF:
				options().gamma = static_cast<double>(hyp(3));
				break;
			case POLYNOMIAL:
				options().coef = static_cast<double>(hyp(3));
				options().degree = static_cast<double>(hyp(4));
				break;
			case SIGMOID:
				options().coef = static_cast<double>(hyp(3));
				options().gamma = static_cast<double>(hyp(4));
				break;
			default:
				break;
		}
		
		return 0;
	}
	
	template <class T>
	int SGD<T>::train(LaspMatrix<T> X, LaspMatrix<int> y){
		if (options().usebias && options().bias != 0) {
			Xin = LaspMatrix<T>::vcat(X, LaspMatrix<T>::ones(X.cols(), 1));
		} else {
			Xin = LaspMatrix<T>::hcat(Xin, X);
		}
		
		Yin = LaspMatrix<T>::hcat(Yin, y);
		
		features = Xin.rows();
		n = Xin.cols();
		int ret_val = train_internal();
		total_data += n;
		
		Xin = LaspMatrix<T>();
		Yin = LaspMatrix<T>();
		
		return ret_val;
	}
	
	template <class T>
	int SGD<T>::retrain(LaspMatrix<T> X, LaspMatrix<int> y){
		iter = 0;
		next_ind = 0;
		total_data = 0;
		bias = 0;
		return train(X, y);
	}
	
	template <class T>
	int SGD<T>::predict(LaspMatrix<T> X, LaspMatrix<int>& output){
		LaspMatrix<T> temp;
		int ret_val = predict_internal(X, temp, false);
		output.copy(temp.template convert<int>());
		return ret_val;
	}
	
	template <class T>
	int SGD<T>::confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
		return -1;
	}
	
	template <class T>
	int SGD<T>::predict_confidence(LaspMatrix<T> X, LaspMatrix<int>& output_predictions , LaspMatrix<T>& output_confidence){
		return -1;
	}
	
	template <class T>
	int SGD<T>::distance(LaspMatrix<T> X, LaspMatrix<T>& output){
		return predict_internal(X, output, true);
	}
	
	template <class T>
	T SGD<T>::score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual){
		return -1;
	}
	
	template <class T>
	T SGD<T>::loss(LaspMatrix<T> y_prob, LaspMatrix<int> y_actual){
		return -1;
	}
	
	template <class T>
	T SGD<T>::loss(LaspMatrix<int> y_pred, LaspMatrix<T> y_prob, LaspMatrix<int> y_actual){
		return -1;
	}
	
	template <class T>
	T SGD<T>::test(LaspMatrix<T> X, LaspMatrix<int> y){
		return -1;
	}
	
	template <class T>
	T SGD<T>::test_loss(LaspMatrix<T> X, LaspMatrix<int> y){
		return -1;
	}
	
	template<class T>
	vector<int> SGD<T>::get_sv() {
		return S;
	}
	
	template<class T>
	int SGD<T>::train_internal(){
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
		
		//Random number generator
#ifdef CPP11
		random_device rd;
		mt19937 mt(rd());
		uniform_int_distribution<int> dist(0, n - 1);
#else
		srand (time(0));
#endif
		
		//Timing Variables
		clock_t baseTime = clock();
		
		//kernel options struct for computing the kernel
		kernel_opt kernelOptions;
		kernelOptions.kernel = options().kernel;
		kernelOptions.gamma = options().gamma;
		kernelOptions.degree = options().degree;
		kernelOptions.coef = options().coef;
		
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
		
		//Output support vectors (reset if iterations is reset)
		if (iter == 0){
			xS = LaspMatrix<T>(0, features, 0.0, n, features, true, false);
			xnormS = LaspMatrix<T>(0, 1, 0.0, n, 1, true, false);
			alphas = LaspMatrix<T>(0, 1, 0.0, n, 1, true, true); //(betas)
			alphas_inds = LaspMatrix<T>(n, 1, -1);
		}
		else {
			xS.resize(xS.cols(), xS.rows(), 0.0, xS.cols() + n, features, true, false);
			xnormS.resize(xnormS.cols(), 1, 0.0, xnormS.cols() + n, 1, true, false);
			alphas.resize(alphas.cols(), 1, 0.0, alphas.cols() + n, 1, true, true); //(betas)
			alphas_inds = LaspMatrix<T>(n, 1, -1);
		}
		
		
		//Move support vector stuff to gpu
		if (gpu) {
			xS.transferToDevice();
			xnormS.transferToDevice();
			alphas.transferToDevice();
		}
		
		int k = options().set_size;
		T lambda = static_cast<T>(1.0 / options().C);
		bool linear = options().kernel == 1;
		
		//int next_ind = 0;
		
		T w_norm = 0;
		
		//Main loop
		for (iter = 1; iter <= options().maxiter; ++iter) {
			//Generate random set for sub-gradient calculation
			LaspMatrix<int> At_ind(k, 1);
			set<int> unique;
			
			for (int i = 0; i < k; ++i) {
				int ind = getrandom(n);
				if (unique.count(ind) > 0) {
					--i;
				} else {
					unique.insert(ind);
					At_ind(i) = ind;
				}
			}
			
			//Gather the set
			LaspMatrix<T> At_x;
			x.gather(At_x, At_ind);
			
			LaspMatrix<T> At_xNorm;
			xNorm.gather(At_xNorm, At_ind);
			
			LaspMatrix<T> At_y;
			y.gather(At_y, At_ind);
			
			LaspMatrix<T> At_kernel;
			LaspMatrix<T> At_dist;
			
			//Compute distances to the hyperplane
			if (next_ind == 0) {
				At_dist = LaspMatrix<T>(k, 1, 0.0);
			}else {
				//Calculate the kernel
				At_kernel.getKernel(kernelOptions, xS, xnormS, At_x, At_xNorm, gpu);
				
				At_kernel.colWiseMult(alphas);
				At_kernel.colSum(At_dist);
				At_dist.rowWiseMult(At_y);
				At_dist.add(bias);
				
				//Free up the kernel
				At_kernel = LaspMatrix<T>();
			}
			
			//Update alphas based on the learning rate
			T eta = options().eta / (lambda * static_cast<T>(iter));
			T etaK = eta / static_cast<T>(k);
			
			//For the linear kernel, just update the vector (w) directly
			if (!linear) {
				alphas.multiply(1.0 - (lambda * eta));
			} else {
				xS.multiply(1.0 - (lambda * eta));
			}
			
			w_norm *= (1.0 - (lambda * eta)) * (1.0 - (lambda * eta));
			
			//Move some things back to the host
			if (gpu) {
				At_dist.transferToHost();
				At_y.transferToHost();
				At_ind.transferToHost();
			}
			
			//Subgradient of the bias
			T biasGrad = 0;
			
			//Find the subset that violates the margin
			for (int i = 0; i < k; ++i) { //Subset index
				if (At_dist(i) < 1.0){
					int ind = At_ind(i); //Original vector index
					int alpha_ind = alphas_inds(ind); //Index in support set
					
					T alpha_update = At_y(i) * etaK;
					biasGrad += At_y(i);
					
					//Compute the update to the regularization term
					LaspMatrix<T> w_kernel;
					vector<int> ind_vec;
					ind_vec.push_back(i);
					
					LaspMatrix<T> x_new = At_x.gather(ind_vec);
					LaspMatrix<T> xNorm_new = At_xNorm.gather(ind_vec);
					
					if(next_ind != 0){
						LaspMatrix<T> w_dist;
						w_kernel.getKernel(kernelOptions, xS, xnormS, x_new, xNorm_new, gpu);
						w_kernel.colWiseMult(alphas);
						
						if (gpu) {
							w_kernel.transferToHost();
						}
						
						w_kernel.colSum(w_dist);
						w_norm += 2.0 * w_dist(0) * alpha_update;
					}
					
					w_kernel.getKernel(kernelOptions, x_new, xNorm_new, x_new, xNorm_new, gpu);
					
					if (gpu) {
						w_kernel.transferToHost();
					}
					
					w_norm += w_kernel(0) * alpha_update * alpha_update;
					
					//If the vector is not a support add it
					if (alpha_ind == -1 && !linear) {
						alpha_ind = next_ind;
						alphas_inds(ind) = alpha_ind;
						++next_ind;
						
						alphas.resize(next_ind, 1);
						xS.resize(next_ind, features);
						xnormS.resize(next_ind, 1);
						
						xS.setCol(alpha_ind, At_x, i);
						xnormS.setCol(alpha_ind, At_xNorm, i);
						S.push_back(total_data + ind);
					}
					
					//Update the weights
					if (gpu && !linear) {
						LaspMatrix<T> alpha_to_update = alphas(alpha_ind, 0, alpha_ind + 1, 1);
						alpha_to_update.add(alpha_update);
					} else if (!linear) {
						alphas(alpha_ind) += alpha_update;
					} else { //Linear kernel
						//Add a blank vector if nothing is there yet
						if (next_ind == 0) {
							xS.resize(1, features);
							xS.multiply(0);
							alphas.resize(1,1);
							alphas.add(1.0);
							S.push_back(1);
							++next_ind;
						}
						
						//Add our new weighted vector to the model
						x_new.multiply(alpha_update);
						xS.add(x_new);
						xS.colSqSum(xnormS);
					}
				}
			}
			
			//Do final alpha update
			T update = w_norm;
			update = (1.0 / sqrt(lambda)) / sqrt(update);
			
			if (update < 1.0) {
				if (!linear) {
					alphas.multiply(update);
				} else {
					xS.multiply(update);
				}
				
				w_norm *= (update * update);
			}
			
			//Update the bias (Not sure if correct)
			if (update > 1) {
				update = 1;
			}
			biasGrad *= (-1.0 / k);
			
			if (options().usebias && options().bias == 0) {
				bias -= eta * update * biasGrad;
			}
			
			//Print status update
			if (iter % 50 == 0 && options().verb > 1) {
				//Calculate the objective value for the set
				T obj = (lambda / 2.0) * w_norm;
				
				At_kernel.getKernel(kernelOptions, xS, xnormS, At_x, At_xNorm, gpu);
				
				At_kernel.colWiseMult(alphas);
				At_kernel.colSum(At_dist);
				At_dist.rowWiseMult(At_y);
				At_dist.add(bias);
				
				if (gpu) {
					At_dist.transferToHost();
				}
				
				T loss = 0.0;
				for (int i = 0; i < k; ++i) {
					loss += (1.0 / k) * max(0.0, 1.0 - At_dist(i));
				}
				
				obj += loss;
				
				double newTime = ((double)(clock() - baseTime)) / CLOCKS_PER_SEC;
				cout << "At iteration: " << iter << ", time: " << newTime << ", support vectors: " << next_ind << ", obj: " << obj << endl;
				cout << "(loss: " << loss << ", regularizer: " << (lambda / 2.0) * w_norm << ", bias: " << bias << ")\n" << endl;
			}
		}
		
		if (gpu) {
			alphas.transferToHost();
			y.transferToHost();
			xS.transferToHost();
		}
						
		if (options().verb > 0){
			cout << "Training Complete" << endl;
		}
		
		return 0;
	}
	
	template<class T>
	int SGD<T>::predict_internal(LaspMatrix<T> X, LaspMatrix<T>& output, bool dist){
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
		
		//Get a matrix of the weights to multiply in later
		LaspMatrix<T> betas = alphas;
		
		//Create data matricies
		LaspMatrix<T> dXS = xS;
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
				Ke.resize(X.cols() / numChunks, xS.cols());
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
			LaspMatrix<T> classes;
			
			betas.multiply(Ke, classes);
			classes.add(bias);
			classes.transferToHost();  //Avoid race condition in operator()
			
			//Set classes
			if(!dist){
#pragma omp parallel for
				for (int i=0; i < chunkSize; ++i) {
					output(i + chunkStart) = classes(i, 0) > 0 ? 1 : -1;
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

#endif /* defined(__SP_SVM__pegasos_model__) */
