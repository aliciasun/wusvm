#ifndef LASP_SVM_TOP_H
#define LASP_SVM_TOP_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas.h>
#endif

using namespace std;

namespace lasp{
  enum errors{ CORRECT, UNOPENED_FILE_ERROR, WRONG_NUMBER_ARGS };
  enum arg { FILE_IN, C_IN, GAMMA_IN, DONE };
  #define LINE_SIZE 1024

  struct svm_node{
    int index;
    double value;
  };

  struct opt{
    int nb_cand;
    int set_size;
    int maxiter;
    double base_recomp;
    int verb;
    int contigify;
    int maxnewbasis;
    double stoppingcriterion;
  };

  struct svm_problem{
    int C, gamma, n, features;
    opt options;
    int * y;
    svm_node** x;
  };


  int load_data(char*, int&, int&, int*&, svm_node**&);
  int lasp_svm( svm_problem );
  int parse_and_load( int, char **, svm_problem&);
  float* sparse_to_full(svm_problem);
  void tempOutputCheck(svm_node**,int*);
  void exit_with_help();
}

#endif
