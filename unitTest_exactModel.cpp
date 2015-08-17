#include "fileIO_new.h"
#include <iostream>


int main(int argc, char* argv[]){
  cout << "========== TESTING IO Functionality ===========" << endl;
  loadTest();
}

int loadTest() {
  std::String file1 = "miniTest.train";
  LaspMatrix<double> Xin;
  LaspMatrix<double> Yin;
  load_LIBSVM(file1.c_str(), Xin, Yin, 5, 5, false);
  Xin.printMatrix("X");
  Yin.printMatrix("Y");
  return 0;
}

/*
int modelTest() {
  char* file1= '/research-projects/mlgroup/datasets/adult/a9a';
  LaspMatrix<double> Xin;
  LaspMatrix<double> Yin;
  loadLibSVM(file1, Xin, Yin, 32561, 123, false);
  SVM_exact svm();
  svm.train(Xin, Yin,); 
  cout << svm.lossPrimal(optimize_options().lambda);
  return 0;
}
*/
