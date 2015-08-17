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

#ifndef LASP_PREDICT_H
#define LASP_PREDICT_H

namespace lasp
{

  //The highest-level classify function.
  //Given a model, a set of sparse data,
  //a reference to where you want the
  //number of correct predictions stored,
  //and a filename where you want your
  //predictions saved, classifies, fills
  //in the number correct, and outputs classifications
  //to a file.
  void classify_host(svm_model& myModel,
		     svm_sparse_data& sparseData,
		     int& correct,
		     char* outputfile);
  
  //Same as classify, but uses a dag model instead of the 1v1 voting
  //algorithm.
  void dagclassify_host(svm_model& myModel,
                        svm_sparse_data& sparseData,
                        int& correct,
                        char* outputfile);

  //Given full svm_model/svm_data structs, puts the
  //classifications, as calcuated from the model, and
  //the total number of correct predictions that
  //were made. Optional boolean argument can be
  //set to true if you want the distances to the hyperplane
  //in int* classifications, rather than the ultimate classifications
  int classify_host(double* classifications,
		    svm_binary_model& myModel,
		    svm_full_data& myData, opt& options,
		    bool distFromHyperplane = false);

  svm_binary_model get_binary_model(svm_model& sparseModel,
				    int less,
				    int greater);

}
#endif
