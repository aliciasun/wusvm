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

#include "options.h"
#include <cmath>

namespace lasp {
    
	opt::opt(){
		//sets parameters to defaults
		nb_cand = 10;
		set_size = 5000;
		maxiter = 20;
		base_recomp = std::pow(2,0.5);
		verb = 1;
		contigify = false;
		maxnewbasis = 800;
		candbufsize = 0;
		stoppingcriterion = 5e-6;
		maxcandbatch = 100;
		coef = 0;
		degree = 2;
		kernel = RBF;
		modelFile = "output.model";
		C = 1;
		gamma = 1; //we will set this later, but the defaut requires knowledge of the dataset.
		plattScale = 0;
		usegpu = false;
		randomize = false;
		single = false;
		pegasos = false;
		usebias = true;
		shuffle = true;
		smallKernel = false;
		bias = 1;
		start_size = 100;
		boIter = 25;
		maxGPUs = 1;
		stopIters = 1;
		optimize = false;
		costSensitive = false;
		unified = false;
		dag = false;
		logHyp = false;
		optimizeParameters = true;
		tau = 1.0;
		scale = 1.0;
		noise = 1.0;
		mean = 0;
		forward_stopping = true;
		eta = 1.0;
		alpha = 1.0;
		beta = 1.0;
		
		
        compressedSVM = false;
        compressedRatio = 1;
	}
	
	kernel_opt opt::kernel_options(){
		kernel_opt kernelOptions;
		kernelOptions.kernel = kernel;
		kernelOptions.gamma = gamma;
		kernelOptions.scale = scale;
		kernelOptions.degree = degree;
		kernelOptions.coef = coef;
		kernelOptions.alpha = alpha;
		kernelOptions.beta = beta;
		return kernelOptions;
	}
	
	kernel_opt::kernel_opt() : kernel(RBF), gamma(1), coef(1), degree(1), scale(1) {}
	
	optimize_options::optimize_options() : gpu(false), log(false), maximize(false), tuneLambda(true), optimizeParameters(true), logHyp(true), allIters(false), optCost(false), shuffle(false), maxIter(25), warmupIter(3), numCand(1000), passes(1), grid(UNIFORM), batch(1), test_iters(-1), tau(1.0), noise(1.0), epsilon(1e-4), lambda(.01), constraint(1.0), infeasibleScale(1e10), momentum(0), logFile(""){}
}
