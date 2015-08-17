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

#ifndef LASP_GAUSSIAN_PROCESS_H
#define LASP_GAUSSIAN_PROCESS_H

#include <iterator>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "lasp_matrix.h"
#include <cstdlib>

namespace lasp {
	
	using namespace std;
	
	template<class T>
	int gaussian_process(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& Xcand, LaspMatrix<T>& mean, LaspMatrix<T>& sig, LaspMatrix<T>& ellIn, optimize_options options);
	
	template<class T>
	int gaussian_process(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& Xcand, LaspMatrix<T>& mean, LaspMatrix<T>& sig, optimize_options options);
	
	template<class T>
	T gaussian_process_likelihood(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, LaspMatrix<T>& hypIn, LaspMatrix<T>& der, optimize_options options);

	//Functor for likelihood optimization
	template<class T>
	class likelihood_eval {
		LaspMatrix<T> &X, &Y;
		optimize_options& options;
		
	public:
		likelihood_eval(LaspMatrix<T>& Xin, LaspMatrix<T>& Yin, optimize_options& optin) :
		X(Xin), Y(Yin), options(optin){}
		
		T operator()(LaspMatrix<T>& hyp, LaspMatrix<T>& grad){
			T lik = gaussian_process_likelihood(X, Y, hyp, grad, options);
			grad.negate();
			return -lik;
		}
	};

}


#endif
