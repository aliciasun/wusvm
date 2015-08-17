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

#include "wusvm.h"
#include "bayes_opt.h"
#include "parsing.h"
#include "fileIO.h"
#include "pegasos.h"
#include "getopt.h"
#include "predict.h"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lasp;

int create_data_from_array(wusvm_data* svm_data, double* x, double* y, int features, int points){
	svm_full_data full;
	
	size_t size = (size_t)features * (size_t)points;
	float* x_single = new float[size];
	float* y_single = new float[points];
	std::copy(x, x + size, x_single);
	std::copy(y, y + points, y_single);
	
	full.x = x_single;
	full.y = y_single;
	full.numFeatures = features;
	full.numPoints = points;
	
	svm_sparse_data* data = new svm_sparse_data;
	full_data_to_sparse(*data, full);
	
	delete [] x_single;
	delete [] y_single;
	
	*svm_data = reinterpret_cast<wusvm_data>(data);
	return NO_ERROR;
}


int create_test_data_from_array(wusvm_data* svm_data, double* x, int features, int points){
	svm_full_data full;
	
	float* y = new float[points];
	for (int i = 0; i < points; ++i) {
		y[i] = 1.0;
	}
	
	size_t size = (size_t)features * (size_t)points;
	float* x_single = new float[size];
	std::copy(x, x + size, x_single);
	
	full.x = x_single;
	full.y = y;
	full.numFeatures = features;
	full.numPoints = points;
	
	svm_sparse_data* data = new svm_sparse_data;
	full_data_to_sparse(*data, full);
	
	delete [] y;
	delete [] x_single;
	
	*svm_data = reinterpret_cast<wusvm_data>(data);
	return NO_ERROR;
}


int create_data_from_file(wusvm_data* svm_data, const char* file){
	svm_sparse_data* data = new svm_sparse_data;
	
	int error = load_sparse_data(file, *data);
	if (error) {
		delete data;
		return FILE_ERROR;
	}
	
	*svm_data = reinterpret_cast<wusvm_data>(data);
	return NO_ERROR;
}


int get_labels_from_data(wusvm_data svm_data, int* labels){
	svm_sparse_data* data = reinterpret_cast<svm_sparse_data*>(svm_data);
	
	if (!data) {
		return TYPE_ERROR;
	}
	
	if(!labels){
		return UNALLOCATED_POINTER_ERROR;
	}
	
	if (data->outputClassifications.size() == 0) {
		return NOT_FIT_ERROR;
	}
	
	int index = 0;
	for (vector<int>::iterator iter = data->outputClassifications.begin();
			iter != data->outputClassifications.end(); ++iter, ++index) {
		labels[index] = *iter;
	}
	
	return NO_ERROR;
}


int free_data(wusvm_data svm_data){
	svm_sparse_data* data = reinterpret_cast<svm_sparse_data*>(svm_data);
	
	if (!data) {
		return TYPE_ERROR;
	}
	
	delete data;
	return NO_ERROR;
}


int create_options(wusvm_options* svm_options){
	*svm_options = reinterpret_cast<wusvm_options>(new opt);
	return NO_ERROR;
}


int set_kernel(wusvm_options svm_options, wusvm_kernel kernel){
	opt* options = reinterpret_cast<opt*>(svm_options);
	
	if (!options) {
		return TYPE_ERROR;
	}
	
	options->kernel = kernel;
	return NO_ERROR;
}


int set_options(wusvm_options svm_options, int optCount, char** optArgs){
	opt* options_ptr = reinterpret_cast<opt*>(svm_options);
	
	if (!options_ptr) {
		return TYPE_ERROR;
	}
	
	opt& options = *options_ptr;

	//Variables we need for parsing arguments
	float floatVal;
	int intVal;
	int c;
	char* end;
	
	static struct option long_options[] = {
		{"gpu", no_argument, 0, 'u'},
		{"version", no_argument, 0, 'q'},
		{"random", no_argument, 0, 'f'},
		{"single", no_argument, 0, 'l'},
		{"float", no_argument, 0, 'l'},
		{"pegasos", no_argument, 0, 'p'},
		{"backtracking", required_argument, 0, 'i'},
		{"bias", required_argument, 0, 'o'},
		{"omp_threads", required_argument, 0, 'T'},
		{"noshuffle", no_argument, 0, 'e'},
		{"no_cache", no_argument, 0, 'K'},
		{"dag", no_argument, 0, 'D'},
		{"contigify_kernel", no_argument, 0, 't'},
		{"maxgpus", required_argument, 0, 'y'}
	};
	
	
	while((c = getopt_long(optCount, optArgs, "n:s:i:y:b:v:t:m:x:a:pj:g:c:k:r:d:o:huqflw:eS:KT:ODI", long_options, 0)) != -1)
		switch(c)
	{
		case 'n':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.nb_cand = intVal;
			break;
		case 's':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.set_size = intVal;
			break;
		case 'i':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.maxiter = intVal;
			break;
		case 'b':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.plattScale = intVal;
			break;
		case 'v':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				return ARGUMENT_ERROR;
			options.verb = intVal;
			break;
		case 't':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 1)
				return ARGUMENT_ERROR;
			options.contigify = intVal;
			break;
		case 'm':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.maxnewbasis = intVal;
			break;
		case 'T':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal <= 0)
				return ARGUMENT_ERROR;
#ifdef _OPENMP
			omp_set_num_threads(intVal);
#endif
			break;
		case 'x':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0')
				return ARGUMENT_ERROR;
			options.stoppingcriterion = floatVal;
			break;
		case 'g':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0')
				return ARGUMENT_ERROR;
			options.gamma = floatVal;
			break;
		case 'c':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0')
				return ARGUMENT_ERROR;
			options.C = floatVal;
			break;
		case 'k':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				return ARGUMENT_ERROR;
			options.kernel = intVal;
			break;
		case 'I':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				return ARGUMENT_ERROR;
			options.boIter = intVal;
			break;
		case 'r':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0' || floatVal < 0)
				return ARGUMENT_ERROR;
			options.coef = floatVal;
			break;
		case 'd':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0' || floatVal < 0)
				return ARGUMENT_ERROR;
			options.degree = floatVal;
			break;
		case 'o':
			options.usebias = true;
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0' || floatVal < 0)
				return ARGUMENT_ERROR;
			options.bias = floatVal;
			break;
		case 'a':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.maxcandbatch = intVal;
			break;
		case 'j':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.start_size = intVal > 4 ? intVal : 0 ;
			break;
		case 'y':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.maxGPUs = intVal;
			break;
		case 'S':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				return ARGUMENT_ERROR;
			options.stopIters = intVal;
			break;
		case 'u':
			options.usegpu = true;
			break;
		case 'f':
			options.randomize = true;
			break;
		case 'p':
			options.pegasos = true;
			break;
		case 'l':
			options.single = true;
			break;
		case 'e':
			options.shuffle = false;
			break;
		case 'K':
			options.smallKernel = true;
			break;
		case 'O':
			options.optimize = true;
			break;
		case 'D':
			options.dag = true;
			break;
		case 'q':
			version();
			std::exit(0);
			break;
		case 'h':
			return ARGUMENT_ERROR;
		case '?':
			return ARGUMENT_ERROR;
		default:
			return ARGUMENT_ERROR;
	}
	
	return NO_ERROR;
}


int set_option_int(wusvm_options svm_options, char option, int value){
	opt* options_ptr = reinterpret_cast<opt*>(svm_options);
	
	if (!options_ptr) {
		return TYPE_ERROR;
	}
	
	opt& options = *options_ptr;
	
	int intVal = 0;
	
	switch(option)
	{
		case 'n':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.nb_cand = intVal;
			break;
		case 's':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.set_size = intVal;
			break;
		case 'i':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.maxiter = intVal;
			break;
		case 'b':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.plattScale = intVal;
			break;
		case 'v':
			intVal = value;
			if(intVal < 0 || intVal > 4)
				return ARGUMENT_ERROR;
			options.verb = intVal;
			break;
		case 't':
			intVal = value;
			if(intVal < 0 || intVal > 1)
				return ARGUMENT_ERROR;
			options.contigify = intVal;
			break;
		case 'm':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.maxnewbasis = intVal;
			break;
		case 'T':
			intVal = value;
			if(intVal <= 0)
				return ARGUMENT_ERROR;
#ifdef _OPENMP
			omp_set_num_threads(intVal);
#endif
			break;
		case 'k':
			intVal = value;
			if(intVal < 0 || intVal > 4)
				return ARGUMENT_ERROR;
			options.kernel = intVal;
			break;
		case 'I':
			intVal = value;
			if(intVal < 0 || intVal > 4)
				return ARGUMENT_ERROR;
			options.boIter = intVal;
			break;
		case 'a':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.maxcandbatch = intVal;
			break;
		case 'j':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.start_size = intVal > 4 ? intVal : 0 ;
			break;
		case 'y':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.maxGPUs = intVal;
			break;
		case 'S':
			intVal = value;
			if(intVal < 0)
				return ARGUMENT_ERROR;
			options.stopIters = intVal;
			break;
		case 'u':
			options.usegpu = value != 0;
			break;
		case 'f':
			options.randomize = value != 0;
			break;
		case 'p':
			options.pegasos = value != 0;
			break;
		case 'l':
			options.single = value != 0;
			break;
		case 'e':
			options.shuffle = value != 0;
			break;
		case 'K':
			options.smallKernel = value != 0;
			break;
		case 'O':
			options.optimize = value != 0;
			break;
		case 'D':
			options.dag = value != 0;
			break;
		case 'h':
			return ARGUMENT_ERROR;
		case '?':
			return ARGUMENT_ERROR;
		default:
			return ARGUMENT_ERROR;
	}
	
	return NO_ERROR;

}


int set_option_double(wusvm_options svm_options, char option, double value){
	opt* options_ptr = reinterpret_cast<opt*>(svm_options);
	
	if (!options_ptr) {
		return TYPE_ERROR;
	}
	
	opt& options = *options_ptr;
	
	double floatVal = 0;
	
	switch(option)
	{
		case 'x':
			options.stoppingcriterion = value;
			break;
		case 'g':
			options.gamma = value;
			break;
		case 'c':
			options.C = value;
			break;
		case 'r':
			floatVal = value;
			if(floatVal < 0)
				return ARGUMENT_ERROR;
			options.coef = floatVal;
			break;
		case 'd':
			floatVal = value;
			if(floatVal < 0)
				return ARGUMENT_ERROR;
			options.degree = floatVal;
			break;
		case 'o':
			options.usebias = true;
			floatVal = value;
			if(floatVal < 0)
				return ARGUMENT_ERROR;
			options.bias = floatVal;
			break;
		default:
			return ARGUMENT_ERROR;
	}
	
	return NO_ERROR;
}


int free_options(wusvm_options svm_options){
	opt* options = reinterpret_cast<opt*>(svm_options);
	
	if (!options) {
		return TYPE_ERROR;
	}
	
	delete options;
	return NO_ERROR;
}


int fit_model(wusvm_model* model, wusvm_data svm_data, wusvm_options svm_options){
	opt* options_ptr = reinterpret_cast<opt*>(svm_options);
	svm_sparse_data* data_ptr = reinterpret_cast<svm_sparse_data*>(svm_data);
	
	if (!options_ptr) {
		return TYPE_ERROR;
	}

	if (!data_ptr) {
		return TYPE_ERROR;
	}
	
	opt& options = *options_ptr;
	svm_sparse_data& myData = *data_ptr;
	
	
	svm_time_recorder recorder;

	if (options.optimize) {
		optimize_parameters<double>(myData, options);
	}
	
	vector<svm_problem> solvedProblems;
	//tracks holdout data, which is used for platt scaling later.
	vector<svm_sparse_data> holdouts;
	
	for(int i = 0; i < myData.orderSeen.size(); ++i) {
		for(int j = i+1; j < myData.orderSeen.size(); ++j) {
			int firstClass = myData.orderSeen[i];
			int secondClass = myData.orderSeen[j];
			
			if(options.verb > 0) {
				cout << "Training " << firstClass << " v. " << secondClass << " classifier" << endl;
			}
			
			if(options.plattScale) {
				svm_sparse_data holdoutData;
				svm_problem curProblem = get_onevsone_subproblem(myData,
																			 holdoutData,
																			 firstClass,
																			 secondClass,
																			 options);
				if (options.single) {
					if (options.pegasos) {
						pegasos_svm_host<float>(curProblem);
					} else {
						lasp_svm_host<float>(curProblem);
					}
					
				} else {
					if (options.pegasos) {
						pegasos_svm_host<double>(curProblem);
					} else {
						lasp_svm_host<double>(curProblem);
					}
				}
				
				solvedProblems.push_back(curProblem);
				holdouts.push_back(holdoutData);
			}
			else {
				svm_problem curProblem = get_onevsone_subproblem(myData,
																			 firstClass,
																			 secondClass,
																			 options);
				if (options.single) {
					if (options.pegasos) {
						pegasos_svm_host<float>(curProblem);
					} else {
						lasp_svm_host<float>(curProblem);
					}
					
				} else {
					if (options.pegasos) {
						pegasos_svm_host<double>(curProblem);
					} else {
						lasp_svm_host<double>(curProblem);
					}
				}
				
				solvedProblems.push_back(curProblem);
			}
		}
	}
	
	//This could be better
	svm_model myModel = get_model_from_solved_problems(solvedProblems, holdouts, myData.orderSeen);
	*model = reinterpret_cast<wusvm_model>(new svm_model(myModel));
	
	return NO_ERROR;
}


int save_model(wusvm_model model, const char* file){
	svm_model* model_ptr = reinterpret_cast<svm_model*>(model);
	
	if (!model_ptr) {
		return TYPE_ERROR;
	}
	
	int error = write_model(*model_ptr, file);
	
	if (error) {
		return FILE_ERROR;
	}
	
	return NO_ERROR;
}


int load_model(wusvm_model* model, const char* file){
	svm_model* myModel = new svm_model;
	
	int error = load_model(file, *myModel);
	
	if (error) {
		delete myModel;
		return FILE_ERROR;
	}
	
	*model = reinterpret_cast<wusvm_model>(myModel);
	return NO_ERROR;
}


int free_model(wusvm_model model){
	svm_model* myModel = reinterpret_cast<svm_model*>(model);
	
	if (!myModel) {
		return TYPE_ERROR;
	}
	
	delete myModel;
	return NO_ERROR;
}


int classify_data(wusvm_data test_data, wusvm_model model, wusvm_options svm_options){
	opt* options_ptr = reinterpret_cast<opt*>(svm_options);
	svm_sparse_data* data_ptr = reinterpret_cast<svm_sparse_data*>(test_data);
	svm_model* model_ptr = reinterpret_cast<svm_model*>(model);
	
	if (!model_ptr) {
		return TYPE_ERROR;
	}
	
	if (!options_ptr) {
		return TYPE_ERROR;
	}
	
	if (!data_ptr) {
		return TYPE_ERROR;
	}
	
	opt& options = *options_ptr;
	svm_sparse_data& sparseData = *data_ptr;
	svm_model& myModel = *model_ptr;
	
	myModel.options = options;
	
	int correct;
	if (options.dag) {
		dagclassify_host(myModel, sparseData, correct, 0);
	} else {
		classify_host(myModel, sparseData, correct, 0);
	}
	
	if(options.verb > 2){
		cout << "Accuracy = " << (correct*1.0)/sparseData.numPoints * 100 << "% ";
		cout << "(" << correct << "/" << sparseData.numPoints << ") ";
		cout << "(classification)" << endl;
	}
	
	return NO_ERROR;
}
