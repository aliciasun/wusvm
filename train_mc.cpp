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

#include "lasp_func.h"
#include "bayes_opt.h"
#include "parsing.h"
#include "fileIO.h"
#include "pegasos.h"
#include "optimize.h"
#include "logistic_model.h"
#include "exact_model.h"
#include "SVEN_model.h"
#include <iostream>
#include "fileIO_new.h"
#include "time.h"

int loadTest() {
  const char* file1= "/Users/chipschaff/Documents/wustl/AdvancedMachineLearning/WU_SVM/test_data/easy.train";
  lasp::LaspMatrix<double> Xin;
  lasp::LaspMatrix<double> Yin;
  int n = 38;
  int p = 2;
  lasp::load_LIBSVM(file1, Xin, Yin, n, p, false);
  Xin.printMatrix("X");
  Yin.printMatrix("Y");
  return 0;
}


int modelTest(char* file1, int n, int p, int Tee, int L) {
  //char* file1= "/research-projects/mlgroup/datasets/adult/easy.train";
  //const char* file1= "/Users/chipschaff/Documents/wustl/AdvancedMachineLearning/WU_SVM/test_data/australian_scale";

  lasp::LaspMatrix<double> Xin;
  lasp::LaspMatrix<int> Yin;
  //int n = 463715;//690;//32561;//38;//32561;
  //int p = 90;//14;//123;//2;//123;
  lasp::load_LIBSVM(file1, Xin, Yin, n, p, false);
  cout << "I have loaded my datas" << endl;
  //lasp::SVEN_model<double> svm(Tee,L);
  lasp::SVM_exact<double> svm;
    //Yin = -1*Yin;
    //Yin.printMatrix("Yin");
  time_t t = time(0);
  clock_t t2 = clock();
  //lasp::LaspMatrix<int> YinInt = Yin.convert<int>();
  svm.train(Xin, Yin);
  int return_time = (int)difftime(time(0),t);
  cout << "TOTAL TIME: "<< difftime(time(0),t) << endl;
  cout << "TOTAL CLOCK: "<< ((float)clock()-t2) / CLOCKS_PER_SEC << endl; 
  cout << "I have trained long and hard" << endl;
  //cout << svm.lossPrimal() << endl;
  //lasp::LaspMatrix<int> output;
  //svm.predict(Xin, output);
  //cout << svm.score(output, Yin) << endl;
  return return_time;
}

int modelTest(){
  return modelTest("/home/research/kolkinn/gabe_test/YMSD_norm.txt", 463751,90,30,30.87);
}

int main(int argc, char* argv[]){
  
  char* datasets[] = {"/home/research/kolkinn/gabe_test/australian_scale","/research-projects/mlgroup2/for_quan/data/MITFaces_norm.txt","/research-projects/mlgroup2/for_quan/data/Yahoo_norm.txt","/home/research/kolkinn/gabe_test/YMSD_norm.txt"};
  int Ns[] = {690,489410,141397,4463751};
  int Ps[] = {14,361,519,90};
  int numD = 4;

  int Ts[] = {3,10,30,100};
  int numT = 4;

  float Ls[] = {30.87,60.4}; 
  int numL = 2;

  
  lasp::LaspMatrix<double> times(numT*numL,numD,0.0);
  /*
  for (int d = 0; d < numD; ++d){
    for (int i=0; i<numT; ++i){
      for (int j=0; j<numL; ++j){
	times(i*numL+j,d) = (double) modelTest(datasets[d],Ns[d],Ps[d],Ts[i],Ls[j]);
      }
    }
  }*/

  times(0,0) = (double) modelTest(datasets[0],Ns[0],Ps[0],Ts[2],Ls[0]);
  

  times.printMatrix("TIMES");
  //modelTest();
  cout << endl << endl;

}







  /* lasp::SVM_exact<double> exact;
     lasp::SVEN_model<double> sven = lasp::SVEN_model<double>(1.0);	
   lasp::opt options;
	lasp::svm_time_recorder recorder;
	lasp::svm_sparse_data myData;
	if(lasp::parse_and_load(argc, argv, options, myData) != lasp::CORRECT){
		return 1;
	}
	
	if (options.optimize) {
		 lasp::optimize_parameters<double>(myData, options);
	}
	
	vector<lasp::svm_problem> solvedProblems;
	//tracks holdout data, which is used for platt scaling later.
	vector<lasp::svm_sparse_data> holdouts;
	
	for(int i = 0; i < myData.orderSeen.size(); ++i) {
		for(int j = i+1; j < myData.orderSeen.size(); ++j) {
			int firstClass = myData.orderSeen[i];
			int secondClass = myData.orderSeen[j];
			
			if(options.verb > 0) {
				cout << "Training " << firstClass << " v. " << secondClass << " classifier" << endl;
			}
			
			if(options.plattScale) {
				lasp::svm_sparse_data holdoutData;
				lasp::svm_problem curProblem = lasp::get_onevsone_subproblem(myData,
																			 holdoutData,
																			 firstClass,
																			 secondClass,
																			 options);
				if (options.single) {
					if (options.pegasos) {
						lasp::pegasos_svm_host<float>(curProblem);
					} else {
						lasp::lasp_svm_host<float>(curProblem);
					}
					
				} else {
					if (options.pegasos) {
						lasp::pegasos_svm_host<double>(curProblem);
					} else {
						lasp::lasp_svm_host<double>(curProblem);
					}
				}
				
				solvedProblems.push_back(curProblem);
				holdouts.push_back(holdoutData);
			}
			else {
				lasp::svm_problem curProblem = lasp::get_onevsone_subproblem(myData,
																			 firstClass,
																			 secondClass,
																			 options);
				if (options.single) {
					if (options.pegasos) {
						lasp::pegasos_svm_host<float>(curProblem);
					} else {
						lasp::lasp_svm_host<float>(curProblem);
					}
					
				} else {
					if (options.pegasos) {
						lasp::pegasos_svm_host<double>(curProblem);
					} else {
						lasp::lasp_svm_host<double>(curProblem);
					}
				}
				
				solvedProblems.push_back(curProblem);
			}
		}
	}
	
	lasp::svm_model myModel = lasp::get_model_from_solved_problems(solvedProblems, holdouts, myData.orderSeen);
	lasp::write_model(myModel, options.modelFile.c_str());
  */
