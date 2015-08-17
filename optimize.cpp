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

#include "optimize.h"


namespace lasp {
	
	template<class T>
	T rosenbrock_test(LaspMatrix<T>& x, LaspMatrix<T>& grad){
		int dim = x.rows();
		
		LaspMatrix<T> x_lower = x(0,0,1,dim-1);
		LaspMatrix<T> x_upper = x(0,1,1,dim);
		
		//Compute gradient
		grad.resize(1, dim);
		grad.multiply(0.0);
		
		LaspMatrix<T> grad_lower = grad(0,0,1,dim-1);
		LaspMatrix<T> grad_upper = grad(0,1,1,dim);
		
		grad_lower.copy(x_lower);
		grad_lower.pow(2);
		grad_lower.negate();
		grad_lower.add(x_upper);
		grad_lower.eWiseMultM(x_lower, grad_lower);
		grad_lower.multiply(-400.0);
		grad_lower.add(x_lower);
		grad_lower.add(x_lower);
		grad_lower.subtract(2.0);
		
		LaspMatrix<T> grad_upTemp(1, dim-1, 0.0);
		grad_upTemp.copy(x_lower);
		grad_upTemp.pow(2.0);
		grad_upTemp.negate();
		grad_upTemp.add(x_upper);
		grad_upTemp.multiply(200.0);
		grad_upper.add(grad_upTemp);
		
		//Compute function value
		LaspMatrix<T> fVal(1, dim-1, 0.0);
		fVal.copy(x_lower);
		fVal.pow(2.0);
		fVal.negate();
		fVal.add(x_upper);
		fVal.pow(2.0);
		fVal.multiply(100.0);
		
		LaspMatrix<T> tempVal(1, dim-1, 0.0);
		tempVal.copy(x_lower);
		tempVal.negate();
		tempVal.add(1.0);
		tempVal.pow(2.0);
		
		fVal.add(tempVal);
		fVal.colSum(tempVal);
		return tempVal(0);
	}
	
	template<class T>
	T getRandFloat(T min, T max){
		T r = 0.0;
#ifdef CPP11
		default_random_engine generator;
		uniform_real_distribution<T> dist(min, max);
		r = dist(generator);
#else
		r = static_cast <T> (rand() / (static_cast <T> (RAND_MAX / (max - min)))) + min;
#endif
		return r;
	}
	
	int getRandInt(int min, int max){
		int r = 0.0;
#ifdef CPP11
		default_random_engine generator;
		uniform_int_distribution<int> dist(min, max);
		r = dist(generator);
#else
		r = (rand() % (max - min)) + min;
#endif
		return r;
	}
	
	//Uniformly random samples from our feature space
	template<class T>
	int set_parameter_grid_uniform(LaspMatrix<T> min, LaspMatrix<T> max, LaspMatrix<T>& grid, int numCand){
		grid = LaspMatrix<T>(numCand, min.rows());
		for (int col = 0; col < numCand; ++col) {
			for (int row = 0; row < min.rows(); ++row) {
				grid(col, row) = getRandFloat(min(row), max(row));
			}
		}
		
		return 0;
	}
	
	//Fixed grid of samples from our feature space (numCands specifies granularity of each dimension)
	template<class T>
	int set_parameter_grid_fixed(LaspMatrix<T> min, LaspMatrix<T> max, LaspMatrix<T>& grid, LaspMatrix<int> numCands){
		int numCand = 0;
		int dim = min.rows();
		LaspMatrix<T> steps(1, dim);
		
		for(int i = 0; i < dim; ++i){
			numCand *= numCands(i);
			steps(i) = (max(i) - min(i)) / static_cast<T>(numCands(i));
		}
		
		//Counter for index in every dimension
		vector<int> counters(dim, 0);
		
		//Output matrix
		grid = LaspMatrix<T>(numCand, dim);
		
		//Loop through all candidate points
		for (int i = 0; i < numCand; ++i) {
			//Update indicies
			for(int curDim = 0; curDim < dim; ++curDim){
				counters[curDim]++;
				if (counters[curDim] >= numCands(curDim)) {
					counters[curDim] = 0;
				} else {
					break;
				}
			}
			
			//Add point
			LaspMatrix<T> newPoint(1, dim);
			for (int curDim = 0; curDim < dim; ++curDim) {
				newPoint(curDim) = min(curDim) + (steps(curDim) * counters[curDim]);
			}
			
			grid.setCol(i, newPoint);
		}
		
		return 0;
	}
	
	//"Square" grid of samples from our feature space
	template<class T>
	int set_parameter_grid_square(LaspMatrix<T> min, LaspMatrix<T> max, LaspMatrix<T>& grid, int numCand){
		double dim = static_cast<double>(min.rows());
		int dimSize = static_cast<int>(std::pow(numCand, 1.0 / dim));
		LaspMatrix<int> numCands(1, dim, dimSize);
		return set_parameter_grid_fixed(min, max, grid, numCands);
 	}
	
	template<class T>
	int expected_improvement(LaspMatrix<T>& Yin, LaspMatrix<T>& mean, LaspMatrix<T>& sig, LaspMatrix<T>& ei, optimize_options options){
		//Find the best current input point
		T Ybest = Yin.minElem();
		
		//Now compute the expected improvement for each candidate
		LaspMatrix<T> Z, Zcdf, Zpdf;
		mean.subtract(Ybest);
		mean.negate();
		mean.eWiseDivM(sig, Z);
		
		Z.normPDF(Zpdf);
		Z.normCDF(Zcdf);
		
		Z.eWiseMultM(Zcdf, Z);
		Z.add(Zpdf);
		sig.eWiseMultM(Z, ei);
				
		return 0;
	}
	
	template<class T>
	int remove_infeasable(LaspMatrix<T>& Yin, LaspMatrix<T>& Yi, optimize_options options){
		int numValues = Yin.rows();
		for (int j = (options.optCost ? 2 : 1); j < numValues; ++j) {
			int constraintNum = j - 1 - (options.optCost ? 1 : 0);
			
			T constraint = static_cast<T>(options.constraint);
			if (options.allConstraints.size() > constraintNum) {
				constraint = static_cast<T>(options.allConstraints[constraintNum]);
			}
			
			for (int k = 0; k < Yi.cols(); ++k) {
				if (Yin(k, j) > constraint) {
					Yi(k) = options.infeasibleScale; //Large value to simulate infeasibility
				}
			}
		}
		
		return 0;
	}
	
	template<class T>
	int compute_best(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& Xcand, optimize_options options){
		LaspMatrix<T> scores, valMean, valSig; //valMean/Sig are mean and sd for value GP (used in EI/cost computation)
		int numValues = Yin.rows();
		
		for (int i = 0; i < numValues; ++i) {
			LaspMatrix<T> mean, sig, hyp;
			LaspMatrix<T> Yi = Yin(0, i, Yin.cols(), i+1).copy();
						
			hyp = LaspMatrix<T>::zeros(1, Xin.rows() + 2);
			hyp(0) = std::log(0.1);
			
			//Compute maximum log-likelihood for parameters
			if(options.optimizeParameters){
				optimize_options likelihood_options;
				likelihood_eval<T> evaluator(Xin, Yi, options);
								
				conjugate_optimize(evaluator, hyp, likelihood_options);
				
			}
			gaussian_process(Xin, Yi, Xcand, mean, sig, hyp, options);
						
			//We assume that the first row is the score, second is the
			// cost (only if specified in options.optCost) and all others
			// are constraints
			if (i == 0) {
				valMean = mean;
				valSig = sig;
				remove_infeasable(Yin, Yi, options);
				expected_improvement(Yi, mean, sig, scores, options);
			} else if (i == 1 && options.optCost){ //Cost GP
				
				LaspMatrix<T> Y0 = Yin(0, 0, Yin.cols(), 1);
				T bestVal = Y0.minElem();
				
				LaspMatrix<T> prWorse(mean.cols(), 1, bestVal);
				prWorse.normCDF(valMean, valSig);
				prWorse.negate();
				prWorse.add(1.0);
				
				mean.eWiseMultM(prWorse, mean);
				mean.add(1.0);
				
				scores.eWiseDivM(mean, scores);
				
			} else { //Constraint GP
				int constraintNum = i - 1 - (options.optCost ? 1 : 0);
				
				T constraint = static_cast<T>(options.constraint);
				if (options.allConstraints.size() > constraintNum) {
					constraint = static_cast<T>(options.allConstraints[constraintNum]);
				}
				
				LaspMatrix<T> prFeas(mean.cols(), 1, constraint);
				LaspMatrix<T> prFeasUpper(mean.cols(), 1, 0.0);
				prFeas.normCDF(mean, sig);
				prFeasUpper.normCDF(mean, sig);
				prFeas.subtract(prFeasUpper);
				
				scores.eWiseMultM(prFeas);
			}
		}
		
		//Find and return the best expected improvement among the candidates
		int row, maxInd;
		T best = scores.maxElem(maxInd, row);
				
		if (options.log) {
			cout << "Candidate: " << maxInd << ", Expected improvement: " << best << endl;
		}
		
		//If we have early stopping on and EI negative, signal to stop
		if (!options.allIters && best < 0) {
			maxInd = -1;
		}
		
		return maxInd;
	}
	
	template double rosenbrock_test(LaspMatrix<double>& x, LaspMatrix<double>& grad);
	template float rosenbrock_test(LaspMatrix<float>& x, LaspMatrix<float>& grad);
	
	template double getRandFloat(double min, double max);
	template float getRandFloat(float min, float max);
	
	template int set_parameter_grid_uniform(LaspMatrix<double> min, LaspMatrix<double> max, LaspMatrix<double>& grid, int numCand);
	template int set_parameter_grid_uniform(LaspMatrix<float> min, LaspMatrix<float> max, LaspMatrix<float>& grid, int numCand);
	
	template int set_parameter_grid_fixed(LaspMatrix<double> min, LaspMatrix<double> max, LaspMatrix<double>& grid, LaspMatrix<int> numCands);
	template int set_parameter_grid_fixed(LaspMatrix<float> min, LaspMatrix<float> max, LaspMatrix<float>& grid, LaspMatrix<int> numCands);
	
	template int set_parameter_grid_square(LaspMatrix<double> min, LaspMatrix<double> max, LaspMatrix<double>& grid, int numCand);
	template int set_parameter_grid_square(LaspMatrix<float> min, LaspMatrix<float> max, LaspMatrix<float>& grid, int numCand);
	
	template int expected_improvement(LaspMatrix<double>& Yin, LaspMatrix<double>& mean, LaspMatrix<double>& sig, LaspMatrix<double>& ei, optimize_options options);
	template int expected_improvement(LaspMatrix<float>& Yin, LaspMatrix<float>& mean, LaspMatrix<float>& sig, LaspMatrix<float>& ei, optimize_options options);
	
	template int remove_infeasable(LaspMatrix<double>& Yin, LaspMatrix<double>& Yi, optimize_options options);
	template int remove_infeasable(LaspMatrix<float>& Yin, LaspMatrix<float>& Yi, optimize_options options);
	
	template int compute_best(LaspMatrix<double>& Xin, LaspMatrix<double>& Yin, LaspMatrix<double>& Xcand, optimize_options options);
	template int compute_best(LaspMatrix<float>& Xin, LaspMatrix<float>& Yin, LaspMatrix<float>& Xcand, optimize_options options);
}
