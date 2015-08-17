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

#ifndef LASP_OPTIMIZE_H
#define LASP_OPTIMIZE_H

#include "gaussian_process.h"
#ifdef CPP11
#include <random>
#include <cmath>
#endif

namespace lasp {
	
	using namespace std;
	
	/*
	Optimization functions (Implementations below):
	 
	template<class FUNC, class T>
	int gradient_optimize(FUNC eval, LaspMatrix<T>& x, optimize_options options);
	 
	template<class FUNC, class T>
	int newton_optimize(FUNC eval, LaspMatrix<T>& X, optimize_options options);
	
	template<class FUNC, class T>
	int conjugate_optimize(FUNC eval, LaspMatrix<T>& X, optimize_options options);
	 
	template<class FUNC, class T>
	int stochastic_optimize(FUNC eval, LaspMatrix<T>& x, LaspMatrix<T> Xin, LaspMatrix<T> Yin, optimize_options options);
	
	template<class FUNC, class T>
	int bayes_optimize(FUNC eval, LaspMatrix<T>& x, LaspMatrix<T> min, LaspMatrix<T> max, optimize_options options);
	 
	 
	Testing functions:
	 
	template<class FUNC, class T>
	T check_gradient(FUNC eval, LaspMatrix<T>& x, T e);
	*/
	
	template<class T>
	T rosenbrock_test(LaspMatrix<T>& x, LaspMatrix<T>& grad);
	
	
	//Helper functions:
	template<class T>
	T getRandFloat(T min, T max);
	
	int getRandInt(int min, int max);
	
	template<class T>
	int set_parameter_grid_uniform(LaspMatrix<T> min, LaspMatrix<T> max, LaspMatrix<T>& grid, int numCand);
	
	template<class T>
	int set_parameter_grid_fixed(LaspMatrix<T> min, LaspMatrix<T> max, LaspMatrix<T>& grid, LaspMatrix<int> numCands);
	
	template<class T>
	int set_parameter_grid_square(LaspMatrix<T> min, LaspMatrix<T> max, LaspMatrix<T>& grid, int numCand);
	
	template<class T>
	int expected_improvement(LaspMatrix<T>& Yin, LaspMatrix<T>& mean, LaspMatrix<T>& sig, LaspMatrix<T>& ei, optimize_options options);
	
	template<class T>
	int remove_infeasable(LaspMatrix<T>& Yin, LaspMatrix<T>& Yi, optimize_options options);
	
	template<class T>
	int compute_best(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& Xcand, optimize_options options);
	
	
	//Simple gradient descent
	template<class FUNC, class T>
	int gradient_optimize(FUNC eval, LaspMatrix<T>& x, optimize_options options){
		bool log = options.log;
		bool logToFile = !(options.logFile.empty());
		
		//Open a file for output
		ofstream stream;
		if (logToFile && log) {
			stream.open(options.logFile.c_str());
			if(!stream.is_open()){
				logToFile = false;
				log = false;
			}
		}
		
		bool max = options.maximize;
		T epsilon = static_cast<T>(options.epsilon);
		T lambda = static_cast<T>(options.lambda) * (max ? 1 : -1);
		T diff = epsilon + 1;
		
		LaspMatrix<T> grad;
		T value = eval(x, grad);
		T oldValue = value;
		
		for (int iter = 0; iter < std::abs(options.maxIter) && diff > epsilon; ++iter) {
			grad.multiply(lambda);
			x.add(grad);
			
			oldValue = value;
			value = eval(x, grad);
			
			if (options.tuneLambda && (max ? value < oldValue : value > oldValue)) {
				lambda *= .5;
			}
			
			LaspMatrix<T> diffMat;
			grad.colSqSum(diffMat);
			
			diff = diffMat(0,0);
			
			//Log current iteration
			if (logToFile && log) {
				stream << value;
				for (int gradIter = 0; gradIter < grad.size(); ++gradIter){
					stream << ", " << grad(gradIter);
				}
				stream << "\n";
			} else if (log) {
				cout << "Gradient descent iteration: " << iter << " complete, value: " << value << ", x: ";
				for (int gradIter = 0; gradIter < x.size(); ++gradIter){
					if(gradIter > 0) cout << ", ";
					cout << x(gradIter);
				}
				cout << endl;
			}
		}
		
		return 0;
	}
	
	template<class FUNC, class T>
	int gradient_optimize(FUNC eval, LaspMatrix<T>& x){
		return gradient_optimize(eval, x, optimize_options());
	}
	
	
	//Newton's method
	template<class FUNC, class T>
	int newton_optimize(FUNC eval, LaspMatrix<T>& x, optimize_options options){
		bool log = options.log;
		bool logToFile = !(options.logFile.empty());
		
		//Open a file for output
		ofstream stream;
		if (logToFile && log) {
			stream.open(options.logFile.c_str());
			if(!stream.is_open()){
				logToFile = false;
				log = false;
			}
		}
		
		bool max = options.maximize;
		T epsilon = static_cast<T>(options.epsilon);
		T lambda = static_cast<T>(options.lambda) * (max ? 1 : -1);
		T diff = epsilon + 1;
		
        	LaspMatrix<T> diffMat, grad, hess, step;
		T value = eval(x, grad, hess);
		cout << "VALUE :" << value << endl;
            	grad = t(grad);
		grad.colSqSum(diffMat);
            	grad = t(grad);
            	diff = diffMat(0,0);
	    	cout << "DIFF: " << diff << endl;
		std::cin.ignore();
		T oldValue = value;
        	int count;

		for (int iter = 0; iter < std::abs(options.maxIter) && diff > epsilon; ++iter) {
            bool lineSearch = true;
            //lambda = -.01;
            count = 0;
            while(lineSearch && count < 1){
                lineSearch = false;
                grad = t(grad);
                step = t(step);
                hess.solve(grad, step);
                step.multiply(lambda);
                step=t(step);
                x.add(step);
            
            
                oldValue = value;
            
                grad = t(grad);
                value = eval(x, grad, hess);

			
                //cout << oldValue <<endl;
                //cout << value << endl;
                //cout << lambda << endl;
                if (options.tuneLambda && (max ? value < oldValue : value > oldValue)) {
                    lineSearch = true;
                    //cout << "NOooooooooooooooooooo" << endl;
                    lambda *= .5;
                    step.multiply(-1);
                    x.add(step);
                    value = eval(x,grad,hess);
                    
                } else if (options.tuneLambda) {
                    lambda *= 1.05;
                }
                count++;
                cout << "============== " << count << " =============" << endl << endl;
            }
            cout << "END LINE SEARCH" << endl;
            cout << endl;
            grad = t(grad);
			grad.colSqSum(diffMat);
            grad = t(grad);
            diff = diffMat(0,0);
	    cout << "DIFF: " << diff << endl;
            //diff = 0;
            //grad.printMatrix("grad");
            //for (int i = 0; i < grad.cols(); i++) {
            //    diff += grad(i,0)*grad(i,0);
            //}
			
			
			//Log current iteration
			if (logToFile && log) {
				stream << value;
				for (int gradIter = 0; gradIter < grad.size(); ++gradIter){
					stream << ", " << grad(gradIter);
				}
				stream << "\n";
			} else if (log) {
				cout << "Newton's method iteration: " << iter << " complete, value: " << value << ", x: ";
				for (int gradIter = 0; gradIter < x.size(); ++gradIter){
					if(gradIter > 0) cout << ", ";
					cout << x(gradIter);
				}
				cout << endl;
			}
		}
        if (diff > epsilon) {
            cout << "MAX ITER REACHED" << endl;
        }
        cout << epsilon << endl;
        cout << diff << endl;
        //x.printMatrix("x");
		return 0;
	}
	
	template<class FUNC, class T>
	int newton_optimize(FUNC eval, LaspMatrix<T>& x){
		return newton_optimize(eval, x, optimize_options());
	}
	
	//Absolutely shameless copy of Carl Edward Rasmussen's minimize.m fuction
	//Add citation or something for this
	template<class FUNC, class T>
	int conjugate_optimize(FUNC eval, LaspMatrix<T>& X, optimize_options options){
		bool log = options.log;
		bool logToFile = !(options.logFile.empty());
		
		//Open a file for output
		ofstream stream;
		if (logToFile && log) {
			stream.open(options.logFile.c_str());
			if(!stream.is_open()){
				logToFile = false;
				log = false;
			}
		}
		
		bool doMax = options.maximize;
		
		T INT = 0.1;
		T EXT = 3.0;
		int MAX = 20;
		T RATIO = 10;
		T SIG = 0.1;
		T RHO = SIG/2;
		
		T red = 1.0;
		bool lineSearch = options.maxIter > 0;
		int length = std::abs(options.maxIter);
		
		int i = 0;
		bool ls_failed = false;
		
		LaspMatrix<T> df0;
		T f0 = eval(X, df0);
		if (doMax) {
			f0 = -f0;
			df0.negate();
		}
		
		if (!lineSearch) i++;
		
		LaspMatrix<T> s, d0Mat;
		df0.negate(s);
		df0.multiply(s, d0Mat, false, true);
        //df0.multiply(s, d0Mat, true, false);
        if (d0Mat.size() != 1) cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		T d0 = d0Mat(0);
		T x3 = red / (1.0 - d0);
		
		//Predeclare a whole bunch of variables
		T F0, x2, f2, d2, f3, d3, x1, f1, d1, A, B, x4, f4, d4;
		int M;
		LaspMatrix<T> X0, dF0, df3;
		
		int iter = 0;
				
		while(i < length){
			if (lineSearch) i++;
			X0.copy(X);
			dF0.copy(df0);
			F0 = f0;
			
			M = lineSearch ? MAX : std::min(MAX, length-i);
			
			while(true){
				df3.copy(df0);
				x2 = 0.0; f2 = f0; d2 = d0; f3 = f0;
				
				bool success = false;
				while (!success && M > 0) {
					try {
						M--;
						if (!lineSearch) i++;
						
						LaspMatrix<T> evalX, evalS;
						s.multiply(x3, evalS);//okay
						X.add(evalS, evalX);
						f3 = eval(evalX, df3);
						
						if (doMax) {
							f3 = -f3;
							df3.negate();
						}
						
						//Add error check here
						
						success = true;
						
					} catch (...) {
						x3 = (x2 + x3) / 2.0;
					}
				}
				
				if(f3 < F0) {
					LaspMatrix<T> tempS;
					s.multiply(x3, tempS);
					X.add(tempS, X0);
					
					F0 = f3;
					dF0.copy(df3); //Check this
				}
				
				LaspMatrix<T> d3Mat;
                df3.multiply(s, d3Mat, false, true);
                //df3.multiply(s, d3Mat, true, false);
                if (d3Mat.size() != 1) {cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;}
				d3 = d3Mat(0);
				
				if (d3 > SIG * d0 || f3 > f0+x3*RHO*d0 || M == 0) {
					break;
				}
				
				x1 = x2; f1 = f2; d1 = d2;
				x2 = x3; f2 = f3; d2 = d3;
				A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
				B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
								
				if(A != 0){
                    x3 = x1-d1 * (std::pow(x2-x1, static_cast<T>(2.0))) / (B + std::sqrt(B*B-A*d1*(x2-x1)));
				}
					
				if(x3 != x3 || A == 0 || x3 < 0){
					x3 = x2*EXT;
				} else if( x3 > x2*EXT){
					x3 = x2*EXT;
				} else if( x3 < x2+INT*(x2-x1)){
					x3 = x2+INT*(x2-x1);
				}
			}
			
			while ((abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0){
				
				if (d3 > 0 || f3 > f0+x3*RHO*d0){
					x4 = x3; f4 = f3; d4 = d3;
				} else {
					x2 = x3; f2 = f3; d2 = d3;
				}
				
				bool isZero = false;
				if (f4 > f0){
					isZero = (f4-f2-d2*(x4-x2)) == 0;
					if(!isZero){
						x3 = x2 - (0.5*d2*std::pow((x4-x2), static_cast<T>(2.0))) / (f4-f2-d2*(x4-x2));
					}
				} else {
					A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);
					B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
					
					isZero = A == 0;
					if(!isZero){
						x3 = x2+(std::sqrt(B*B-A*d2*std::pow((x4-x2), static_cast<T>(2.0)))-B)/A;
					}
				}
				
				if(x3 != x3 || isZero) {
					x3 = (x2+x4)/2;
				}
				
				x3 = std::max(std::min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));
				
				LaspMatrix<T> evalX, evalS;
				s.multiply(x3, evalS);
				X.add(evalS, evalX);
				f3 = eval(evalX, df3);
				if (doMax) {
					f3 = -f3;
					df3.negate();
				}
								
				if(f3 < F0) {
					LaspMatrix<T> tempS;
					s.multiply(x3, tempS);
					X.add(tempS, X0);
					
					F0 = f3;
					dF0.copy(df3); //Check this
				}

				M--;
				if (!lineSearch) i++;
				
				LaspMatrix<T> d3Mat;
				df3.multiply(s, d3Mat, false, true);
                //df3.multiply(s, d3Mat, true, false);
                if (d3Mat.size() != 1) {cout << "'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''" << endl;}
				d3 = d3Mat(0);
				
			}

			if(std::abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0){
				LaspMatrix<T> tempS;
				s.multiply(x3, tempS);
				X.add(tempS);
				
				f0 = f3;
				
				//Log current iteration
				T value = f0;
				if (logToFile && log) {
					stream << value;
					stream << ", " << i;
					stream << "\n";
				} else if (log) {
					cout << "CG iteration: " << iter << " complete, value: " << value << ", x: ";
					for (int gradIter = 0; gradIter < X.size(); ++gradIter){
						if(gradIter > 0) cout << ", ";
						cout << X(gradIter);
					}
					cout << endl;
				}
				
				LaspMatrix<T> df3_df3, df0_df3, df0_df0;
				df3.multiply(df3, df3_df3, false, true);
				df0.multiply(df3, df0_df3, false, true);
				df0.multiply(df0, df0_df0, false, true);
                //df3.multiply(df3, df3_df3, true, false);
                //df0.multiply(df3, df0_df3, true, false);
                //df0.multiply(df0, df0_df0, true, false);
                if (df0_df3.size() != 1) {cout << "'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''" << endl;}
				T sMul = (df3_df3(0) - df0_df3(0)) / (df0_df0(0));
				s.multiply(sMul);
				s.subtract(df3);
				
				df0.copy(df3);
				d3 = d0;
				
				LaspMatrix<T> d0Mat;
				df0.multiply(s, d0Mat, false, true);
                //df0.multiply(s, d0Mat, true, false);
                if (d0Mat.size() != 1) {cout << "-----------------------------------------------------------------------------------------------" << endl;}
				d0 = d0Mat(0);
				
				if(d0 > 0){
					df0.negate(s);
					LaspMatrix<T> negS;
					s.negate(negS);
					negS.multiply(s, d0Mat, false, true);
                    //negS.multiply(s, d0Mat, true, false);
                    if (d0Mat.size() != 1) {cout << "11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" << endl;}
					d0 = d0Mat(0);
				}
				
				x3 = x3 * std::min(RATIO, d3/(d0-std::numeric_limits<T>::epsilon()));
				
				ls_failed = false;
			} else {
				X.copy(X0);
				df0.copy(dF0);
				f0 = F0;
				
				if(ls_failed || i > length){
					if(log) cerr << "Line search failed, stopping at iteration: " << i << endl;
					break;
				}
				
				df0.negate(s);
                LaspMatrix<T> negS;
                s.negate(negS);
				negS.multiply(s, d0Mat, false, true);
                //negS.multiply(s, d0Mat, true, false);
                if (d0Mat.size() != 1) {cout << "==============================================================================================" << endl;}
				d0 = d0Mat(0);
				
				x3 = 1/(1-d0);
				ls_failed = true;
			}
			++iter;
		}
		
		return 0;
	}
	
	template<class FUNC, class T>
	int conjugate_optimize(FUNC eval, LaspMatrix<T>& X){
		return conjugate_optimize(eval, X, optimize_options());
	}
	
	//Stochasitc gradient descent
	template<class FUNC, class T>
	int stochastic_optimize(FUNC eval, LaspMatrix<T>& x, LaspMatrix<T> Xin, LaspMatrix<T> Yin, optimize_options options) {
		LaspMatrix<int> shuffle_map, revert_map;
		
		//Shuffle the data if requested
		if (options.shuffle) {
			Xin.shuffle(revert_map, shuffle_map);
			Yin.contigify(shuffle_map);
		}
		
		bool log = options.log;
		bool logToFile = !(options.logFile.empty());
		
		//Open a file for output
		ofstream stream;
		if (logToFile && log) {
			stream.open(options.logFile.c_str());
			if(!stream.is_open()){
				logToFile = false;
				log = false;
			}
		}
		
		bool max = options.maximize;
		T epsilon = static_cast<T>(options.epsilon);
		T lambda = static_cast<T>(options.lambda) * (max ? 1 : -1);
		T diff = epsilon + 1;
		
		int batch_size = options.batch;
		int iters = std::ceil(Xin.cols() / (double)batch_size) * options.passes;
		
		LaspMatrix<T> grad;
		int batch = 0, batch_end = 0, next_ind = 0, n = Xin.cols(), feat = Xin.rows(), pass = 0;
		T momentum = static_cast<T>(options.momentum);
		
		batch = std::min(n - next_ind, batch_size);
		batch_end = next_ind + batch;
		
		LaspMatrix<T> X_batch = Xin(next_ind, 0, batch_end, feat);
		LaspMatrix<T> Y_batch = Xin(next_ind, 0, batch_end, 1);
		
		next_ind = batch_end >= n ? 0 : batch_end;
		pass += next_ind == 0 ? 1 : 0;
		
		T value = eval(x, grad, X_batch, Y_batch);
		T oldValue = value;
		
		//Evaluate on full data
		if (options.test_iters > 0) {
			LaspMatrix<T> test_grad;
			oldValue = eval(x, test_grad, Xin, Yin);
		}
		
		for (int iter = 1; iter < iters && diff > epsilon; ++iter) {
			grad.multiply(lambda);
			x.add(grad);
			
			batch = std::min(n - next_ind, batch_size);
			batch_end = next_ind + batch;
			
			X_batch = Xin(next_ind, 0, batch_end, feat);
			Y_batch = Xin(next_ind, 0, batch_end, 1);
			
			next_ind = batch_end >= n ? 0 : batch_end;
			pass += next_ind == 0 ? 1 : 0;
			
			LaspMatrix<T> temp_grad;
			value = eval(x, temp_grad, X_batch, Y_batch);
			
			temp_grad.multiply(1 - momentum);
			grad.multiply(momentum);
			grad.add(temp_grad);
			
			if (options.tuneLambda) {
				lambda /= iter;
			}
			
			if (options.test_iters > 0 && iter % options.test_iters == 0){
				LaspMatrix<T> test_grad;
				T newValue = eval(x, test_grad, Xin, Yin);
				diff = newValue - oldValue;
				oldValue = newValue;
				
				//Log current iteration
				if (logToFile && log) {
					stream << value;
					for (int gradIter = 0; gradIter < grad.size(); ++gradIter){
						stream << ", " << grad(gradIter);
					}
					stream << "\n";
				} else if (log) {
					cout << "Stochastic gradient descent iteration: " << iter << " complete, value: " << oldValue << ", x: ";
					for (int gradIter = 0; gradIter < x.size(); ++gradIter){
						if(gradIter > 0) cout << ", ";
						cout << x(gradIter);
					}
					cout << endl;
				}
			}
		}
		
		//Revert the shuffle
		if (options.shuffle) {
			Xin.revert(revert_map);
			Yin.revert(revert_map);
		}
		
		return 0;
	}
	
	template<class FUNC, class T>
	int stochastic_optimize(FUNC eval, LaspMatrix<T>& x){
		return gradient_optimize(eval, x, optimize_options());
	}
	
	template<class FUNC, class T>
	int bayes_optimize(FUNC eval, LaspMatrix<T>& x, LaspMatrix<T> min, LaspMatrix<T> max, optimize_options options){
		bool log = options.log;
		bool logToFile = !(options.logFile.empty());
		
		//Open a file for output
		ofstream stream;
		if (logToFile && log) {
			stream.open(options.logFile.c_str());
			if(!stream.is_open()){
				logToFile = false;
				log = false;
			}
		}
		
		int parameters = x.rows();
		int numValues = 1;
		
		int numIter = std::abs(options.maxIter);
		
		//Intialize our candidate points and our starting point for the search
		LaspMatrix<T> Xcand;
		LaspMatrix<T> Ysolved; //Wait to allocate until we know output size
		LaspMatrix<T> Xsolved(1, parameters, 0.0, numIter + 1, parameters);
		Xsolved.copy(x);
		
		//Set our parameter search space
		if (options.grid == FIXED) {
			set_parameter_grid_square(min, max, Xcand, options.numCand);
		} else {
			set_parameter_grid_uniform(min, max, Xcand, options.numCand);
		}
			
		//Start our bayes opt loop
		for (int iter = 0; iter < numIter; ++iter) {
			if (iter > 0) {
				int nextCand = 0;
				
				if (iter < options.warmupIter) {
					nextCand = getRandInt(0, Xcand.cols());
				} else {
					//Pick the next point and add it to our list of tried points
					nextCand = compute_best(Xsolved, Ysolved, Xcand, options);
				}
				
				if (nextCand == -1){
					break;
				}
				
				Xsolved.resize(iter + 1, parameters);
				Ysolved.resize(iter + 1, numValues);
				Xsolved.setCol(iter, Xcand, nextCand);
				
				//Remove the chosen one from the list of candidates
				vector<int> keepCand;
				for(int k = 0; k < Xcand.cols(); ++k){
					if (k != nextCand) {
						keepCand.push_back(k);
					}
				}
				
				LaspMatrix<T> candTemp;
				Xcand.gather(candTemp, keepCand);
				Xcand = candTemp;
			}
			
			//Set our parameters for this run
			LaspMatrix<T> runParams;
			LaspMatrix<T> runParamsTemp = Xsolved(Xsolved.cols() - 1, 0, Xsolved.cols(), parameters);
			runParams.copy(runParamsTemp);
			
			LaspMatrix<T> value;
			value = eval(runParams);
			
			//Allocate Ysolved
			if(Ysolved.size() == 0){
				numValues = value.size();
				Ysolved = LaspMatrix<T>(1, numValues, 0.0, numIter + 1, numValues);
			}
			
			//Log current iteration
			if (logToFile && log) {
				stream << value(0);
				for (int paramIter = 0; paramIter < parameters; ++paramIter){
					stream << ", " << runParams(paramIter);
				}
				stream << endl;
			} else if (log) {
				cout << "Bayes opt iteration: " << iter << " complete, value: " << value(0) << ", parameters: ";
				for (int paramIter = 0; paramIter < parameters; ++paramIter){
					if(paramIter > 0) cout << ", ";
					cout << runParams(paramIter);
				}
				cout << endl;
			}
			
			//The rest of the function is setup for minimization, so just flip the sign of the value
			// Note: this should not affect cost or constraints
			if (options.maximize) {
				value(0) *= -1;
			}
			
			Ysolved.setCol(Ysolved.cols() - 1, value);
		}
		
		int row, minInd;
		
		LaspMatrix<T> xTemp = Xsolved(minInd, 0, minInd + 1, parameters);
		x.copy(xTemp);
		
		return 0;
	}
	
	template<class FUNC, class T>
	int bayes_optimize(FUNC eval, LaspMatrix<T>& x, LaspMatrix<T> min, LaspMatrix<T> max){
		return bayes_optimize(eval, x, min, max, optimize_options());
	}
	
	//Check that eval give the correct gradient for its function (result should be very small)
	template<class FUNC, class T>
	T check_gradient(FUNC eval, LaspMatrix<T>& x, T e){
		int dim = x.rows();
		
		LaspMatrix<T> grad, tempGrad;
		T value = eval(x, grad);
		
		LaspMatrix<T> dh(1, dim, 0.0);
		for(int i = 0; i < dim; ++i){
			x(i) += e;
			T y2 = eval(x, tempGrad);
			x(i) -= 2*e;
			T y1 = eval(x, tempGrad);
			x(i) += e;
			dh(i) = (y2 - y1) / (2*e);
		}
		
		LaspMatrix<T> diff, sum, diffNorm, sumNorm;
		dh.subtract(grad, diff);
		dh.add(grad, sum);
		
		diff.colSqSum(diffNorm);
		sum.colSqSum(sumNorm);
		
		return std::sqrt(diffNorm(0)) / std::sqrt(sumNorm(0));
	}
	
	template<class FUNC, class T>
	T check_gradient(FUNC eval, LaspMatrix<T>& x){
		return check_gradient(eval, x, 1e-4);
	}

}


#endif
