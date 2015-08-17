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

#include "svm.h"
#include "predict.h"
#include "kernels.h"
#include "fileIO.h"
#include <cstdlib>
#include "parsing.h"
#include "getopt.h"

int main(int argc, char** argv)
{
	lasp::svm_model myModel;
	lasp::svm_sparse_data sparseData;
		
	lasp::opt& options = myModel.options;
	options.verb = 1;
	options.usegpu = false;
	options.maxGPUs = 1;
	options.single = false;
	options.smallKernel = false;
	
	
	int optCount = argc;
	char ** optArgs = argv;
	
	char* modelFile = 0;
	char* dataFile = 0;
	char* outputFile = 0;
	bool dag = false;
	
	static struct option long_options[] = {
		{"gpu", no_argument, 0, 'u'},
		{"version", no_argument, 0, 'q'},
		{"single", no_argument, 0, 'l'},
		{"float", no_argument, 0, 'l'},
		{"omp_threads", required_argument, 0, 'T'},
		{"no_cache", no_argument, 0, 'K'},
		{"dag", no_argument, 0, 'D'},
		{"help", no_argument, 0, 'h'},
		{"maxgpus", required_argument, 0, 'y'}
	};
	
	//Variables we need for parsing arguments
	float floatVal;
	int intVal;
	int c;
	char* end;
	
	while((c = getopt_long(optCount, optArgs, "Dy:v:s:huqlKT:", long_options, 0)) != -1)
		switch(c)
	{
		case 'v':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0 || intVal > 4)
				lasp::exit_classify();
			options.verb = intVal;
			break;
		case 's':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				lasp::exit_classify();
			options.set_size = intVal;
			break;
		case 'T':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal <= 0)
				lasp::exit_classify();
#ifdef _OPENMP
			omp_set_num_threads(intVal);
#endif
			break;
		case 'y':
			intVal = strtol(optarg, &end, 10);
			if(end == optarg || *end != '\0' || intVal < 0)
				lasp::exit_classify();
			options.maxGPUs = intVal;
			break;
		case 'u':
			options.usegpu = true;
			break;
		case 'l':
			options.single = true;
			break;
		case 'K':
			options.smallKernel = true;
			break;
		case 'D':
			dag = true;
			break;
		case 'q':
			lasp::version();
			std::exit(0);
			break;
		case 'h':
			lasp::exit_classify();
		case '?':
			lasp::exit_classify();
		default:
			lasp::exit_classify();
	}
	
	//now deal with non options, aka files.
	//just take the first non-option argument
	//and assume that it is the filename.
	if(optind < optCount){
		dataFile = optArgs[optind];
	} else {
		lasp::exit_classify();
	}
	
	if(optind + 1 < optCount){
		modelFile = optArgs[optind + 1];
	} else {
		lasp::exit_classify();
	}
	
	if(optind + 2 < optCount){
		outputFile = optArgs[optind + 2];
	} else {
		lasp::exit_classify();
	}

	
	int status;
	status = lasp::load_model(modelFile, myModel);
	if(status != 0) { cout << "model loading error" << endl; std::exit(0);}
	
	status = lasp::load_sparse_data(dataFile, sparseData);
	if(status != 0) { cout << "data loading error" << endl; std::exit(0);}
	
	//Set options that might change from command line
	myModel.options = options;
	
	int correct;
	if (dag) {
		lasp::dagclassify_host(myModel, sparseData, correct, outputFile);
	} else {
		lasp::classify_host(myModel, sparseData, correct, outputFile);
	}
	
	if(options.verb > 0){
		cout << "Accuracy = " << (correct*1.0)/sparseData.numPoints * 100 << "% ";
		cout << "(" << correct << "/" << sparseData.numPoints << ") ";
		cout << "(classification)" << endl;
	}
}
