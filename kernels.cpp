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

#include "kernels.h"
#include "svm.h"

namespace lasp{

template<class T>
LaspMatrix<T> compute_kernel(kernel_opt kernelOptions, LaspMatrix<T> X1, LaspMatrix<T> Xnorm1, int *ind1, int ind1Len, LaspMatrix<T> X2, LaspMatrix<T> Xnorm2, int *ind2, int ind2Len, bool useGPU){
	bool gpu = useGPU;
	LaspMatrix<T> out;
	
	if((X2.data() == 0 && X2.dData() == 0)|| X2.rows() == 0 || X2.cols() == 0){
		LaspMatrix<T> ones(1, ind1Len, 1.0);
		return ones;
	}

	LaspMatrix<T> X1Param = X1, X2Param = X2, Xnorm1Param = Xnorm1, Xnorm2Param = Xnorm2;
	int X1colsParam = X1.cols(), X2colsParam = X2.cols();
	
	if(ind1Len > 0){
		X1colsParam = ind1Len;
		X1Param = LaspMatrix<T>();
		Xnorm1Param = LaspMatrix<T>();
		vector<int> map(ind1, ind1 + ind1Len);
		X1.gather(X1Param, map);
		Xnorm1.gather(Xnorm1Param, map);
	}
	
	if(ind2Len > 0){
		X2colsParam = ind2Len;
		X2Param = LaspMatrix<T>();
		Xnorm2Param = LaspMatrix<T>();
		vector<int> map(ind2, ind2 + ind2Len);
		X2.gather(X2Param, map);
		Xnorm2.gather(Xnorm2Param, map);
	}
	
	if(useGPU){
		out.transferToDevice();
	}
	
	out.getKernel(kernelOptions, X1Param, Xnorm1Param, X2Param, Xnorm2Param, false, false, gpu);
	return out;
}

template LaspMatrix<float> compute_kernel<float>(kernel_opt kernelOptions, LaspMatrix<float> X1, LaspMatrix<float> Xnorm1, int *ind1, int ind1Len, LaspMatrix<float> X2, LaspMatrix<float> Xnorm2, int *ind2, int ind2Len, bool useGPU);
template LaspMatrix<double> compute_kernel<double>(kernel_opt kernelOptions, LaspMatrix<double> X1, LaspMatrix<double> Xnorm1, int *ind1, int ind1Len, LaspMatrix<double> X2, LaspMatrix<double> Xnorm2, int *ind2, int ind2Len, bool useGPU);

}



