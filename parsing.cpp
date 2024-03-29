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

#include "Config.h"
#include "parsing.h"
#include "svm.h"
#include "fileIO.h"
#include "getopt.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void lasp::exit_with_help()
{
	cout << "usage: train_mc [options] training_data_file [model_output_file]\n";
	cout << "options:\n";
	cout << "-k kernel: set type of kernel (default = RBF) \n";
	cout << "\t 0 -- Radial Basis Function: exp(-gamma*|u-v|^2)\n";
	cout << "\t 1 -- Linear: u'*v \n";
	cout << "\t 2 -- Polynomial: (gamma*u'*v + coef)^degree \n";
	cout << "\t 3 -- Sigmoid: tanh(gamma*u'*v + coef)\n";
	cout << "-r coef: sets coefficient value used in some kernel functions (default = 0)\n";
	cout << "-d degree: sets degree value uesd in some kernel functions (default = 3)\n";
	cout << "-g gamma: sets gamma value used in some kernel functions (default = 1 / # of features)\n";
	cout << "-c C: sets c parameter for SVM (default = 1)\n";
	cout << "-n nb_cand: sets nb_cand (default = 10)\n";
	cout << "-s set_size: sets size (default = 5000)\n";
	cout << "-i max_iter: sets max number of backtracking iterations(default = 20)\n";
	cout << "-v verbosity: sets output level (range: 0-3, default = 1)\n";
	cout << "-m max_new_basis: sets maximum number of vectors in new basis (default = 800)\n";
	cout << "-x stopping_criterion: sets the stopping criterion (default = 5e-6)\n";
	cout << "-j jump_set: sets the size of the starting training set (default = 100)\n";
	cout << "-b probability_estimates: use Platt scaling to produce probability estimates\n";
	cout << "-O optimize: use bayesian optimization to set hyperparameters\n";
	cout << "-C cost-sensitive optimize: use BO to set hyperparameters and minimize training time\n";
	cout << "-I opt iterations: number of training runs for hyperparameter optimization (default = 25)\n";
	cout << "--gpu (-u) gpu: uses CUDA to accelerate computation\n";
#ifdef _OPENMP
	cout << "--omp_threads (-T) OpenMP threads: sets the max number of threads to be used by OpenMP\n";
#endif
	cout << "--no_cache (-K) no cache: avoid caching the full kernel matrix\n";
	cout << "--contigify_kernel (-t) contigify kernel: avoid copying full kernel matrix\n";
	cout << "--random (-f) randomize: randomizes training set selection\n";
	cout << "--sgd (-p) SGD classifier: use stochastic gradient descent for training\n";
	cout << "--version (-q) version: displays version number and build details\n";
	cout << "-h help: displays this message\n";
	std::exit(0);
}

void lasp::exit_classify()
{
	cout << "Three arguments required." << endl;
	cout << "usage: ./classify_mc [options] dataFile modelFile outputFile" << endl;
	cout << "options:\n";
	cout << "-v verbosity: sets output level (range: 0-3, default = 1)\n";
	cout << "--gpu (-u) gpu: uses CUDA to accelerate computation\n";
	cout << "--dag (-D) dag: use dag model for multiclass classification\n";
#ifdef _OPENMP
	cout << "--omp_threads (-T) OpenMP threads: sets the max number of threads to be used by OpenMP\n";
#endif
	cout << "--no_cache (-K) no cache: avoid caching the full kernel matrix\n";
	cout << "-s set_size: maximum batch of test points when using -K (default = 5000)\n";
	cout << "--version (-q) version: displays version number and build details\n";
	cout << "-h help: displays this message\n";
	std::exit(0);
}

int lasp::parse_args(int optCount, char **optArgs, lasp::opt &options){
	char* file = 0;
	
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
		{"sgd", no_argument, 0, 'p'},
		{"backtracking", required_argument, 0, 'i'},
		{"bias", required_argument, 0, 'o'},
		{"omp_threads", required_argument, 0, 'T'},
		{"noshuffle", no_argument, 0, 'e'},
		{"no_cache", no_argument, 0, 'K'},
		{"help", no_argument, 0, 'h'},
		{"contigify_kernel", no_argument, 0, 't'},
		{"maxgpus", required_argument, 0, 'y'},
        //(Yu)
        {"compressed", required_argument, 0, 'w'}
	};
	
	
	while((c = getopt_long(optCount, optArgs, "n:s:i:y:bv:tm:x:a:pj:g:c:k:r:d:o:huqflw:eS:KT:OUCI:w", long_options, 0)) != -1)
		switch(c)
	{
		case 'n':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.nb_cand = intVal;
			break;
		case 's':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.set_size = intVal;
			break;
		case 'i':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.maxiter = intVal;
			break;
		case 'b':
			options.plattScale = 1;
			break;
		case 'v':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				exit_with_help();
			options.verb = intVal;
			break;
		case 'm':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.maxnewbasis = intVal;
			break;
		case 'T':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal <= 0)
				exit_with_help();
#ifdef _OPENMP
			omp_set_num_threads(intVal);
#endif
			break;
		case 'x':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0')
				exit_with_help();
			options.stoppingcriterion = floatVal;
			break;
		case 'g':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0')
				exit_with_help();
			options.gamma = floatVal;
			break;
		case 'c':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0')
				exit_with_help();
			options.C = floatVal;
			break;
		case 'k':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				exit_with_help();
			options.kernel = intVal;
			break;
		case 'I':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				exit_with_help();
			options.boIter = intVal;
			break;
		case 'r':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0' || floatVal < 0)
				exit_with_help();
			options.coef = floatVal;
			break;
		case 'd':
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0' || floatVal < 0)
				exit_with_help();
			options.degree = floatVal;
			break;
		case 'o':
			options.usebias = true;
			floatVal = strtof(optarg, &end);
			if(end == optarg || *end != '\0' || floatVal < 0)
				exit_with_help();
			options.bias = floatVal;
			break;
		case 'a':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.maxcandbatch = intVal;
			break;
		case 'j':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.start_size = intVal > 4 ? intVal : 0 ;
			break;
		case 'y':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.maxGPUs = intVal;
			break;
		case 'S':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				exit_with_help();
			options.stopIters = intVal;
			break;
        case 'w':
            options.compressedSVM = true;
			floatVal = strtod(optarg, &end);
            if(end == optarg || *end != '\0' || floatVal <= 0 || floatVal > 1)
				lasp::exit_classify();
			options.compressedRatio = floatVal;
			break;
            
		case 'u':
			options.usegpu = true;
			break;
		case 't':
			options.contigify = true;
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
		case 'C':
			options.optimize = true;
			options.costSensitive = true;
			break;
		case 'U':
			options.unified = true;
			break;
		case 'q':
			version();
			std::exit(0);
			break;
		case 'h':
			exit_with_help();
		case '?':
			exit_with_help();
		default:
			exit_with_help();
	}
	
	//now deal with non options, aka files.
	//just take the first non-option argument
	//and assume that it is the filename.
	if(optind < optCount){
		options.dataFile = optArgs[optind];
	} else {
		exit_with_help();
	}
	
	//Now deal with the model out file
	if(optind +1 < optCount)
		options.modelFile = optArgs[optind+1];
	
	return 0;
}

//parses command line to set parameters and reads in a file in the libsvm format
int lasp::parse_and_load(int optCount,
						 char ** optArgs,
						 opt& options,
						 svm_sparse_data& myData)
{
	parse_args(optCount, optArgs, options);
	
	//loads the data from the file into the svm_sparse_data struct, (Yu)defined in fileIO.cpp
	int error = load_sparse_data(options.dataFile.c_str(), myData);
	
	if(options.gamma == -999 && error == 0) options.gamma = 1.0/myData.numFeatures;
	
	return error;
}

void lasp::version()
{
#ifdef CLASSIFY
	cout << "classify_mc" << endl;
#else
	cout << "train_mc" << endl;
#endif
	cout << "WU-SVM version " << SP_SVM_VERSION_MAJOR << "." << SP_SVM_VERSION_MINOR << endl;
	cout << "\nBuild details:" << endl;
	cout << "\tCUDA acceleration supported: ";
#ifdef CUDA
	cout << "YES" << endl;
#else
	cout << "NO" << endl;
#endif
	cout << "\tMultiple GPUs supported: ";
#ifdef CPP11
	cout << "YES" << endl;
#else
	cout << "NO" << endl;
#endif
	cout << "\tUnified memory supported: ";
#ifdef CUDA6
	cout << "YES" << endl;
#else
	cout << "NO" << endl;
#endif
	cout << "\tOpen MP multithreading supported: ";
#ifdef _OPENMP
	cout << "YES" << endl;
#else
	cout << "NO" << endl;
#endif
	cout << "\tDebug build: ";
#ifndef NDEBUG
	cout << "YES" << endl;
#else
	cout << "NO" << endl;
#endif
	
}
