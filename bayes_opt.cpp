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

#include "bayes_opt.h"
#include "optimize.h"
#include "options.h"
#include "predict.h"
#include "svm.h"
#include "pegasos.h"
#include "fileIO.h"
#include <limits>
#include <algorithm>
#include <set>
#include <ctime>


namespace lasp {
	
	//Functor for evaluating the error
	template<class T>
	class svm_eval{
		svm_sparse_data &trainingData, &holdoutData;
		opt& options;
	public:
		bool dag;
		
		svm_eval(svm_sparse_data &trainingInput, svm_sparse_data &holdoutInput, opt& options_input): trainingData(trainingInput), holdoutData(holdoutInput), options(options_input), dag(false){}
		
		LaspMatrix<T> operator()(LaspMatrix<T> params){
			vector<svm_problem> solvedProblems;
			vector<svm_sparse_data> holdouts;
			
			time_t baseTime = time(0);
			
			//Multiclass loop
			for(int i = 0; i < trainingData.orderSeen.size(); ++i) {
				for(int j = i+1; j < trainingData.orderSeen.size(); ++j) {
					int firstClass = trainingData.orderSeen[i];
					int secondClass = trainingData.orderSeen[j];
					
					if(options.verb > 2) {
						cout << "Training " << firstClass << " v. " << secondClass << " classifier" << endl;
					}
					
					svm_problem curProblem = lasp::get_onevsone_subproblem(trainingData,
																		   firstClass,
																		   secondClass,
																		   options);
					
					//Set the options for solving according to our chosen candidate
					curProblem.options.C = static_cast<double>(params(0));
					
					switch (options.kernel) {
						case RBF:
							curProblem.options.gamma = static_cast<double>(params(1));
							break;
						case SIGMOID:
							curProblem.options.gamma = static_cast<double>(params(1));
							curProblem.options.coef = static_cast<double>(params(2));
							break;
						case POLYNOMIAL:
							curProblem.options.gamma = static_cast<double>(params(1));
							curProblem.options.coef = static_cast<double>(params(2));
							curProblem.options.degree = static_cast<double>(params(3));
							break;
						default:
							break;
					}
					
					if (options.pegasos && options.kernel == LINEAR && options.costSensitive){
						curProblem.options.set_size = static_cast<int>(params(1));
						curProblem.options.maxiter = static_cast<int>(params(2));
					}
					
					//Actually train the svm with the current hyperparameters
					
					
					if (options.pegasos) {
						pegasos_svm_host<T>(curProblem);
					} else {
						lasp_svm_host<T>(curProblem);
					}
										
					solvedProblems.push_back(curProblem);
				}
			}
			
			T newTime = (T)difftime(time(0), baseTime);
			
			//Get the model and classify the holdout data to get the accuracy
			svm_model myModel = get_model_from_solved_problems(solvedProblems, holdouts, trainingData.orderSeen);
			myModel.options = options;
			
			int correct;
			if (dag) {
				dagclassify_host(myModel, holdoutData, correct, 0);
			} else {
				classify_host(myModel, holdoutData, correct, 0);
			}
			
			//Save our accuracy
			double accuracy = (correct*1.0)/holdoutData.numPoints;
			double error = 1.0 - accuracy;
			
			LaspMatrix<T> retVal;
			if (options.costSensitive) {
				retVal = LaspMatrix<T>(1,2,error);
				retVal(1) = newTime;
			} else {
				retVal = LaspMatrix<T>(1,1,error);
			}
			
			return retVal;
		}
		
	};
	
	template<class T>
	int optimize_parameters(svm_sparse_data& myData, opt& options, T tau, bool dag){
		svm_sparse_data holdoutData, trainingData;
		
		//a vector of support vectors and their associated classification
		//in sparse form.
		vector<pair<vector<svm_node>, double> > allData;
		
		int numDataPoints = 0;
		typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
		for(SparseIterator myIter = myData.allData.begin();
			myIter != myData.allData.end();
			++myIter) {
			numDataPoints += myIter->second.size();
			for(int dataPoint = 0; dataPoint < myIter->second.size(); ++dataPoint) {
				pair<vector<svm_node>, double> curVector;
				curVector.first = myIter->second[dataPoint];
				curVector.second = myIter->first;
				allData.push_back(curVector);
			}
		}
		
		//now, we need to shuffle allData
		if (options.shuffle){
			random_shuffle(allData.begin(), allData.end());
		}
		//here, we pop off 30% of the data as "holdout data"
		vector<pair<vector<svm_node>, double> > holdout;
		
		for(int i = 0; i < .3 * allData.size(); ++i) {
			holdout.push_back(allData.back());
			allData.pop_back();
		}
		
		holdoutData.orderSeen = myData.orderSeen;
		holdoutData.numFeatures = myData.numFeatures;
		holdoutData.numPoints = holdout.size();
		holdoutData.multiClass = myData.multiClass;
		//now lets fill up the holdoutData.
		for(int i = 0; i < holdout.size(); ++i) {
			pair<vector<svm_node>, double> curPair = holdout[i];
			holdoutData.allData[int(curPair.second)].push_back(curPair.first);
		}
		
		trainingData.orderSeen = myData.orderSeen;
		trainingData.numFeatures = myData.numFeatures;
		trainingData.numPoints = allData.size();
		trainingData.multiClass = myData.multiClass;
		//now lets fill up the training data.
		for(int i = 0; i < allData.size(); ++i) {
			pair<vector<svm_node>, double> curPair = allData[i];
			trainingData.allData[int(curPair.second)].push_back(curPair.first);
		}
		
		options.shuffle = false;
		
		//Create our evaluation object
		svm_eval<double> evaluator(trainingData, holdoutData, options);
		evaluator.dag = dag;
		
		//Set out initial guess and grid limits
		LaspMatrix<double> Xsolved, Xmin, Xmax;
		switch (options.kernel) {
			case RBF:
				Xsolved.resize(1, 2);
				Xmin.resize(1, 2);
				Xmax.resize(1, 2);
				
				Xsolved(0) = options.C;
				Xmin(0) = 0;
				Xmax(0) = 10;
				
				Xsolved(1) = options.gamma;
				Xmin(1) = 0;
				Xmax(1) = 2;
				break;
			case LINEAR:
				Xsolved.resize(1, 1);
				Xmin.resize(1, 1);
				Xmax.resize(1, 1);
				
				Xsolved(0) = options.C;
				Xmin(0) = 0;
				Xmax(0) = 100;
				
				if (options.pegasos && options.costSensitive) {
					Xsolved.resize(1, 3);
					Xmin.resize(1, 3);
					Xmax.resize(1, 3);
					
					Xsolved(1) = options.set_size;
					Xmin(1) = 1;
					Xmax(1) = 500;
					
					Xsolved(2) = options.maxiter;
					Xmin(2) = 1;
					Xmax(2) = 2500;
				}
	
				break;
			case SIGMOID:
				Xsolved.resize(1, 3);
				Xmin.resize(1, 3);
				Xmax.resize(1, 3);
				
				Xsolved(0) = options.C;
				Xmin(0) = 0;
				Xmax(0) = 10;
				
				Xsolved(1) = options.gamma;
				Xmin(1) = 0;
				Xmax(1) = 2;
				
				Xsolved(2) = options.coef;
				Xmin(2) = 0;
				Xmax(2) = 10;
				break;
			case POLYNOMIAL:
				Xsolved.resize(1, 4);
				Xmin.resize(1, 4);
				Xmax.resize(1, 4);
				
				Xsolved(0) = options.C;
				Xmin(0) = 0;
				Xmax(0) = 10;
				
				Xsolved(1) = options.gamma;
				Xmin(1) = 0;
				Xmax(1) = 2;
				
				Xsolved(2) = options.coef;
				Xmin(2) = 0;
				Xmax(2) = 10;
				
				Xsolved(3) = options.degree;
				Xmin(3) = 0;
				Xmax(3) = 5;
				break;
			default:
				break;
		}
		
		optimize_options opt_opt;
		opt_opt.maxIter = options.boIter;
		//opt_opt.logFile = "bayes_opt_output.txt";
		if (options.verb > 1) {
			opt_opt.log = true;
		}
		
		if (options.costSensitive) {
			opt_opt.optCost = true;
		}
		
		bayes_optimize(evaluator, Xsolved, Xmin, Xmax, opt_opt);
		
		options.C = Xsolved(0);
		
		switch (options.kernel) {
			case RBF:
				options.gamma = Xsolved(1);
				break;
			case SIGMOID:
				options.gamma = Xsolved(1);
				options.coef = Xsolved(2);
				break;
			case POLYNOMIAL:
				options.gamma = Xsolved(1);
				options.coef = Xsolved(2);
				options.degree = Xsolved(3);
				break;
			default:
				break;
		}
		
		if (options.costSensitive && options.pegasos && options.kernel == LINEAR) {
			options.set_size = Xsolved(1);
			options.maxiter = Xsolved(2);
		}
		
		if(options.verb > 0) {
			cout << endl << "Optimization complete! " << endl << "Best parameters are C: " << options.C;
			switch (options.kernel) {
				case LINEAR:
					if (options.costSensitive && options.pegasos && options.kernel == LINEAR) {
						cout << ", set size: " << options.set_size << ", iterations: " << options.maxiter;
					}
					break;
				case RBF:
					cout << ", gamma: " << options.gamma;
					break;
				case SIGMOID:
					cout << ", gamma: " << options.gamma << ", coef: " << options.coef;
					break;
				case POLYNOMIAL:
					cout << ", gamma: " << options.gamma << ", coef: " << options.coef << ", degree: " << options.degree;
					break;
				default:
					break;
			}
			
			cout << endl;
		}
		
		return 0;
	}
	
	
	template int optimize_parameters<double>(svm_sparse_data& myData, opt& options, double tau, bool dag);
	template int optimize_parameters<float>(svm_sparse_data& myData, opt& options, float tau, bool dag);
	
}
