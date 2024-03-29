#ifndef LASP_FILEIO_NEW_H
#define LASP_FILEIO_NEW_H

#include "lasp_matrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


namespace lasp{

  //helper string splitting function
  void split(const string& s, char c, vector<string>& v);

  //split a string based on some delineating character
  vector<string> split_on_delim(const string s, char delim);
  
  //load a libSVM formatted file into an X and Y laspMatrix, possibly determine n and d of training data
  int load_LIBSVM(const char* filename, lasp::LaspMatrix<double>& X, lasp::LaspMatrix<double>& Y, int& n, int& d, bool dimUnknown);

  //load a libSVM formatted file into an X and Y laspMatrix, possibly determine n and d of training data
  int load_LIBSVM(const char* filename, lasp::LaspMatrix<double>& X, lasp::LaspMatrix<int>& Y, int& n, int& d, bool dimUnknown);

}


#endif
