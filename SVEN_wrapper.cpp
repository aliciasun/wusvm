#include <iostream>
#include <fstream>
#include "fileIO.h"
#include "lasp_func.h"
#include "lasp_matrix.h"

int elasticNet_to_SVM(const char* input_file, double t, lasp::LaspMatrix<double>& transformed_data, lasp::LaspMatrix<double>& transformed_labels){

  lasp::svm_sparse_data data_elasticNet;
  
  lasp::load_sparse_data(input_file, data_elasticNet);
 
  int firstClass = data_elasticNet.orderSeen[0];
  int secondClass = data_elasticNet.orderSeen[1];
  lasp::svm_sparse_data holdoutData;
  lasp::opt options;
		
  lasp::svm_problem curProblem = lasp::get_onevsone_subproblem(data_elasticNet, holdoutData, firstClass, secondClass, options);
  
  //Training examples
  lasp::LaspMatrix<double> x = LaspMatrix<double>(p.n, p.features, p.xS).convert<double>();
  
  //Training labels
  lasp::LaspMatrix<double> y = LaspMatrix<double>(p.n,1,p.y).convert<double>();

  //y1 = y./t
  lasp::LaspMatrix<double> y1 = y/t;

  //matrix of ones for copying y1 for each feature
  lasp::LaspMatrix<double> ones_y1 = LaspMatrix<double>(1,p.features,1);

  //y1 copied for each feature
  lasp::LaspMatrix<double> y1_big = ones_y1*y1;

  //self explanatory
  lasp::LaspMatrix<double> x_plus_y1 = x+y1_big;
  lasp::LaspMatrix<double> x_minus_y1 = x-y1_big;

  //Transformed Input Data
  transformed_data = t(hcat(x-y1_big, x+y1_big));
  //transformed_data = t(transformed_data);

  transfomed_labels = hcat(LaspMatrix<double>(p.features,1,1), LaspMatrix<double>(p.features,1,-1));

  //TODO: replace with SUCCESS enum
  return 0;
}

int svm_to_ElasticNet(const char* svm_model_file, int* support_vector_indices, int p, float t, lasp::LaspMatrix<float>& transformed_alphas){
  lasp::svm_model svm_output;
  int status;
  status = lasp::load_model(svm_model_file, svm_output);
  if(status != 0) { cout << "model loading error" << endl; std::exit(0);}
  lasp::LaspMatrix<float> alphas(2*p,1,0);
  for(int i = 0; i < svm_output.numSupportVectors: ++i){
    alphas(support_vector_indices[i]) = svm_output.betas[i];
  }
  
  float sum_a = sum(a);

  //ksi = t / sum(a);
  float ksi = t/sum_a;

  //transformed_alpha = ksi * (alphas(1:p)-alphas(p+1:2p))
  //alternatives: add OMP, use memcpy, create two intermediary matrices and use - operator
  for(int i = 0; i<p; ++i){
    transformed_alphas(i)=alphas(i)-alphas(i+p);
  } 
  transformed_alphas = ksi * transformed_alphas;
  
  //TODO: replace with SUCCESS enum
  return 0;

}
