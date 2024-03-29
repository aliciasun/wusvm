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
#include "parsing.h"
#include "kernels.h"
#include "fileIO.h"
#include <algorithm>
#include "lasp_matrix.h"
#include <cstdlib>
#include <ctime>
#include <cmath>

#ifndef CLASSIFY
#endif
#include "retraining.h"
#include "train_subset.h"
#include "next_point.h"
#include "hessian.h"
#include <string>
#include <cmath>
#include <limits>

#include "gp_model.h"
#include "stopping.h"

#define CPP11
#ifdef CPP11
#include <chrono>
#endif

namespace lasp{
	template<class T>
	int lasp_svm_host(svm_problem& p) {
		int stopIters = 0;
		
		ofstream output_file("GP_TEST.csv");
		output_file << "iteration" << "," << "SVs" << "," << "time" << "," << "error" << "," << "iter mean" << "," << "iter sd" << "," << "SV mean" << "," << "SV sd" << endl;
		FStop<T> sv_stop_model;
		FStop<T> iter_stop_model;
		vector<T> iteration_costs;
		
		srand (time(0));
		bool gpu = p.options.usegpu;
        
		//Might want to move this check elsewhere (i.e. to parsing.cpp)
		if (gpu && DeviceContext::instance()->getNumDevices() < 1) {
			if (p.options.verb > 1) {
				cerr << "No CUDA device found, reverting to CPU-only version" << endl;
			}
			
			p.options.usegpu = false;
			gpu = false;
		} else if(!gpu){
			DeviceContext::instance()->setNumDevices(0);
		} else if (p.options.maxGPUs > -1){
			DeviceContext::instance()->setNumDevices(p.options.maxGPUs);
		}
		
		if (p.options.unified) {
			DeviceContext::instance()->setUseUnified(true);
		}
        
		//Timing Variables
		time_t baseTime = time(0);
		
#ifdef CPP11
		chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
#endif
		
		
		//kernel options struct for computing the kernel
		kernel_opt kernelOptions;
		kernelOptions.kernel = p.options.kernel;
		kernelOptions.gamma = p.options.gamma;
		kernelOptions.degree = p.options.degree;
		kernelOptions.coef = p.options.coef;
		
		//Training examples
		LaspMatrix<T> x = LaspMatrix<double>(p.n, p.features, p.xS).convert<T>();
		
		//Training labels
		LaspMatrix<T> y = LaspMatrix<double>(p.n,1,p.y).convert<T>();
        
		LaspMatrix<T> K;
		
		//Kernel is generally the largest allocation, if its allocation fails, we may need to reduce our set size
        while (p.options.set_size > 0 && !p.options.smallKernel) {
			try {
				if (p.options.start_size > p.options.set_size) {
					p.options.start_size = p.options.set_size;
				}
				
				K.resize(p.n, p.options.set_size + 1, false, false);
			} catch (bad_alloc) {
				p.options.set_size /= 2;
				if (p.options.verb > 1) {
					cerr << "Kernel failed to allocate, reducing max training set to: " << p.options.set_size << endl;
				}
				
				continue;
			}
			break;
		}
		
		//Select iterations for full retraining
		vector<int> R;
		R = retrainIters(p);
		
		//If Dataset is small, prepare to solve it all at once
		bool smallDataSet(false);
		if (p.n < 3000){
			R.clear();
			R.push_back(0);
			R.push_back(p.n);
			smallDataSet=true;
		}
		
        //set the first row of Kernel to be y
		if(!p.options.smallKernel){
			K.setRow(0,y);
		}
		
		LaspMatrix<T> Ksum(1,p.options.set_size + 1,0.0);
		
		{
			for(int i = 0; i < p.n; i++){
				Ksum(0,0) += y(i);
			}
		}
		
		//contains support vectors
		vector<int> S;
		
		//Training examples still incorrectly classified
		vector<int> * erv = new vector<int>();
		
		int out_minus1Length = p.n;
		int xInd = 0;
		
		LaspMatrix<T> HESS(1,1, 0.0,(R.back()+1),(R.back()+1));
		HESS(0,0) = p.options.C * p.n * (1 + 1e-20);
		int ldHESS = (R.back()+1);
		int sizeHESS = 1;
		
		LaspMatrix<T> old_out(1,p.n, 0.0);
		LaspMatrix<T> out(1,p.n, 0.0);
		
		LaspMatrix<T> w(0, 0);
		LaspMatrix<T> x_old(1, R.back()+1, 0.0);
		int x_old_size(0);
		
		//INITIALIZE TRACE VARIABLES: tre, time, obj, betas and bs
		LaspMatrix<T> xNorm (p.n,1, 0.0);
		LaspMatrix<T> xS(0,0,0.0,p.options.set_size,p.features);
		LaspMatrix<T> xnormS(0,0,0.0,p.options.set_size, 1);
		LaspMatrix<T> xerv;
		LaspMatrix<T> yerv;
		LaspMatrix<T> xnormerv;
		LaspMatrix<T> out_minus1(1,p.n);
		
		if (gpu){
			//Check that all gpu allocations succeed
			int error = 0;
			error += x.transferToDevice();
			error += y.transferToDevice();
			error += xNorm.transferToDevice();
			error += xS.transferToDevice();
			error += w.transferToDevice();
			error += out_minus1.transferToDevice();
			error += xnormS.transferToDevice();
			
			if(error != MATRIX_SUCCESS){
				x.transferToHost();
				y.transferToHost();
				xNorm.transferToHost();
				xS.transferToHost();
				w.transferToHost();
				out_minus1.transferToHost();
				xnormS.transferToHost();
				
				if (p.options.verb > 1) {
					cerr << "Device memory insufficient for data, switching to host computation" << endl;
				}
				
				gpu = false;
				p.options.usegpu = false;
				DeviceContext::instance()->setNumDevices(0);
                
			} else {
				error = K.transferToDevice();
				
				if (error != MATRIX_SUCCESS) {
					if (p.options.verb > 1) {
						cerr << "Device memory insufficient for full kernel, leaving on host" << endl;
					}
					
					K.transferToHost();
				}
			}
		}
        
		//Allocating memory and transfering matrices to device
        
		x.colSqSum(xNorm);
		//boolean array to track what's been selected already
		//we will fix candidates to work better later. this is a terrible hack.
		bool alreadySelected[p.n];
		for(int i = 0; i < p.n; ++i){
			alreadySelected[i] = false;
		}
		
		//initialize with 100 random support vectors
		for (vector<int>::iterator r = R.begin(); r!=R.end(); ++r){
			//initialize/reset candidate buffer
			vector<int> candidates;
			int cc = 1;
			//add basis vectors until |S|==r
			int d0 = S.size();
			LaspMatrix<T> cand_K2;
			LaspMatrix<T> cand_K2_norm;
			
			if (gpu){
				cand_K2.transferToDevice();
				cand_K2_norm.transferToDevice();
			}
			
			//Candidate kernel is also typically a large allocation,
			//make sure it succeeds, reducing batch size if needed
			while (!p.options.randomize) {
				try {
					int error = cand_K2.resize(p.options.nb_cand * p.options.maxcandbatch, erv->size(), false, false);
					error += cand_K2_norm.resize(p.options.nb_cand * p.options.maxcandbatch, 1, false, false);
					
					if (error) {
						cand_K2.transferToHost();
						cand_K2_norm.transferToHost();
						continue;
					}
				} catch (bad_alloc) {
					p.options.maxcandbatch /= 2;
					if (p.options.verb > 1) {
						cerr << "Candidate kernel failed to allocate, reducing max batch to: " << p.options.maxcandbatch << endl;
					}
					
					continue;
				}
				break;
			}
			
			for(int d = d0; d < *r; ++d){
				//for small data sets, use everything as basis vector and train
				if (smallDataSet){
					
					for (int i = 0; i < p.n; ++i){
						S.push_back(i);
						alreadySelected[i] = true;
					}
					xS = x.copy();
					xnormS = xNorm.copy();
					break;
					
				}
				
				//Begin by selecting a hundred random basis vectors
				if (*r == p.options.start_size && p.options.start_size != 0){
					int set = p.options.start_size;
				 	xS.resize(set, p.features);
					xnormS.resize(set, 1);
					for(int i = 1; i < set + 1; ++i){
						int randIndex = (i*10) % p.n;
						S.push_back(randIndex);
						alreadySelected[randIndex] = true;
					}
					//gather means to chose S cols from x into xS
					x.gather(xS, S);
					xNorm.gather(xnormS, S);
					break;
				}
                
				//If we have just started or exhausted the previous batch of candidates to consider, get a new batch
				if (cc*10 > candidates.size()){
					//selecting candidate points to select new basis vectors from
					int candstonextretrain = 0;
					for(vector<int>::iterator r = R.begin(); r != R.end(); ++r){
						if (*r > d){
							candstonextretrain = *r - d;
							break;
						}
					}
					
					int candbatchsize = p.options.nb_cand *( min(candstonextretrain, p.options.maxcandbatch));
					d0=S.size();
					candidates.clear();
					for (int i = 1; i <= candbatchsize; ++i){
						candidates.push_back(((i + p.options.nb_cand*d0 - 1) % p.n));
					}
					
					if (!p.options.randomize) {
						//rows of x determined by candidates
						LaspMatrix<T> Xc;
						if (gpu){
							Xc.transferToDevice();
						}
						
						x.gather(Xc,candidates);
						
						//norm of dXc
						LaspMatrix<T> xnormc = LaspMatrix<T>(1,candidates.size());
						if (gpu){
							xnormc.transferToDevice();
						}
						
						xNorm.gather(xnormc,candidates);
						
						cand_K2.getKernel(kernelOptions, xerv, xnormerv, Xc, xnormc, false, false, gpu);
						cand_K2.colWiseMult(yerv);
						cand_K2.colSqSum(cand_K2_norm);
						
					}
					
					cc = 1;
				}
                
				// Which subgroup of the current candidate batch are we working on?
				vector<int> ccI;
				for (int i = (cc-1)*10+1; i <= cc*10; ++i){
					ccI.push_back(i-1);
				}
				
				vector<int> candidatesCCI;
				for (int i = 0; i < ccI.size(); ++i){
					candidatesCCI.push_back(candidates[ccI[i]]);
				}
				
				int select = rand() % ccI.size();
				
				if (!p.options.randomize) {
					
					LaspMatrix<T> cand_K2G;
					LaspMatrix<T> cand_K2normG;
					if (gpu){
						cand_K2G.transferToDevice();
						cand_K2normG.transferToDevice();
					}
					cand_K2.gather(cand_K2G, ccI);
					cand_K2_norm.gather(cand_K2normG, ccI);
                    
					//heuristically choose next support vector
					select = choose_next_point<T>(kernelOptions , candidatesCCI, S, xS, xnormS, x, xNorm, p.features, w, xInd, cand_K2G, ccI.size(), erv->size(), out_minus1, out_minus1Length, p.options.C, p.options.gamma, cand_K2normG, gpu);
				}
				
				int nonSelectIndex = select;
				if(alreadySelected[select]) {
					for(int i = 0; i < p.n;  ++i) {
						if(!alreadySelected[i]) {
							nonSelectIndex = i;
							break;
						}
					}
				}
				
				alreadySelected[nonSelectIndex] = true;
				S.push_back(nonSelectIndex);
				
				xS.resize(d+1, p.features);
				xnormS.resize(d+1, 1);
				
				xS.setCol(S.size()-1,x,S.back());
				xnormS.setCol(S.size()-1,xNorm,S.back());
                
				// Move to next subgroup
				++cc;
			}
			
			int d = S.size();
			d0= sizeHESS - 1;
            
			LaspMatrix<T> dK;
			if (d != d0){
				//gathers the support vectors that are new into their own vector
				vector<int> SG;
				for(int i= S.size()-(d-d0);i < S.size();++i){
					SG.push_back(i);
				}
                
				if (!p.options.smallKernel) {
					K.resize(K.cols(), d+1);
                    
					{
						LaspMatrix<T> K_new = K(0, d0+1, p.n, d+1);
						LaspMatrix<T> xSG, xnormSG;
						
						xS.gather(xSG, SG);
						xnormS.gather(xnormSG, SG);
						K_new.getKernel(kernelOptions, xSG, xnormSG, x, xNorm);
						K_new.rowWiseMult(y);
					}
				}
                
				//calculating Ksum
				LaspMatrix<T> Ksum_new = Ksum(0, d0+1, 1, d+1);
				
				if (p.options.smallKernel) {
					LaspMatrix<T> Kerv = compute_kernel<T>(kernelOptions, x,xNorm, erv->data(), erv->size(), xS,xnormS,SG.data(), SG.size(), gpu);
					
					LaspMatrix<T> yerv_temp;
					y.gather(yerv_temp, *erv);
					Kerv.colWiseMult(yerv_temp);
					
					LaspMatrix<T> kSum_temp;
					Kerv.colSum(kSum_temp);
					kSum_temp.transpose(Ksum_new);
					
				} else {
					LaspMatrix<T> Kerv = K(0, d0+1, p.n, d+1);
                    
					LaspMatrix<T> new_sum;
					LaspMatrix<T> map = Kerv.gatherMap(*erv);
					Kerv.multiply(map, new_sum, false, false, 1.0, 0.0);
					Ksum_new.add(new_sum);
				}
                
				LaspMatrix<double> HESS_in = HESS.template convert<double>(true);
				LaspMatrix<double> K_in = K.template convert<double>();
				LaspMatrix<double> y_in = y.template convert<double>();
				LaspMatrix<double> x_in = x.template convert<double>();
				LaspMatrix<double> xNorm_in = xNorm.template convert<double>();
				
				update_hess(p, HESS_in, K_in, erv, y_in, S, x_in, xNorm_in);
				HESS = HESS_in.convert<T>(true);
				sizeHESS = HESS.rows();
				ldHESS = HESS.mRows();
			}
			
			double obj2 = train_subset_host<T>( p, S, w, HESS, erv, K, Ksum, out, old_out, x_old, x_old_size, x, y, xNorm, xerv, yerv, xnormerv);
			xInd = sizeHESS;
            
			if (obj2<0 || obj2 != obj2){ //Newton steps didn't converge. Probably because the Hessian is not well conditioned. This could be a precision problem.
				if (p.options.verb > 0) {
					cout << "Converge  problem in Newton retraining." << endl;
				}
			}
			
			double newTre = 0;
			
			for(int i = 0; i < p.n; ++i){
				if(out(i) < 0)
					newTre++;
			}
			p.tre.push_back(newTre/p.n);
			
			
			double stopvalue = numeric_limits<double>::max();
			
			if (p.tre.size() > 1){
				stopvalue = - (p.tre[p.tre.size()-1] - p.tre[p.tre.size()-2])/(S.size() - d0);
			}
			
			//Update trace variables
			p.obj.push_back(obj2);
			double newTime = (double)difftime(time(0), baseTime);
			
#ifdef CPP11
			chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
			chrono::duration<double> elapsed_seconds = end-start;
			newTime = elapsed_seconds.count();
#endif
			
			p.time.push_back(newTime);
			vector<double> newBetas;
			newBetas.resize(w.size());
			for(int i = 1; i < w.size(); ++i){
				newBetas[i - 1] = w(i);
			}
			p.betas.push_back(newBetas);
			p.bs.push_back(w(0));
			
			//Output status, if verbose is enabled
			if (p.options.verb > 1 ){
				if (p.options.verb < 3) {
					cout << endl;
				}
				
				cout << "Training Error = " <<  p.tre.back() << ", Time = " << p.time.back() << ", Stopping value = " << stopvalue << "\n\n" << endl;
			} else if (p.options.verb > 0) {
				cout << ". " << flush;
			}
			
			
			for( int i = 0; i< erv->size(); ++i){
				out(i) = out((*erv)[i]);
			}
			
			fill_n(out.data()+erv->size(), p.n-erv->size(),0);
			
			out_minus1.resize(1, erv->size());
            
			LaspMatrix<T> out_erv = out(0,0,1,erv->size());
			out_erv.eWiseOp(out_minus1,-1, 1, 1);
			
			
			if (gpu){
				out_minus1.transferToDevice();
				w.transferToDevice();
			}
            
			//check stopping criterion
			
			//Keep track of the cost of each iteration
			if (p.time.size() == 1) {
				iteration_costs.push_back(p.time.back());
			} else if (p.time.size() > 1){
				iteration_costs.push_back(p.time.back() - p.time[p.time.size() - 2]);
			}
			
			if (p.betas.size() > 1) {
//				//Forward stopping based on gaussian process
				int num_iter = p.betas.size();
				
				LaspMatrix<T> stop_Y(num_iter-1, 1);
				LaspMatrix<T> stop_Xiter(num_iter-1, 1);
				LaspMatrix<T> stop_Xsv(num_iter-1, 1);
				
				//Testing
				for (int iter = 0; iter < num_iter - 1; ++iter) {
					stop_Y(iter) = p.tre[iter];
					stop_Xiter(iter) = iter;
					stop_Xsv(iter) = p.betas[iter].size();
				}
			
				LaspMatrix<T> last_iter;
				last_iter = static_cast<T>(num_iter - 1);
				LaspMatrix<T> last_sv;
				last_sv = static_cast<T>(p.betas.back().size());
				LaspMatrix<T> last_Y;
				last_Y = p.tre.back();
				
				T current_error = last_Y(0);
				T error_thresh = 0.01;
				T error_prob = 0.25;
				
				LaspMatrix<T> sv_mean, sv_sig, cost_mean, cost_sig;
				
				sv_stop_model.train(stop_Xsv(stop_Xsv.cols()-1, 0, stop_Xsv.cols(), stop_Xsv.rows()), stop_Y(stop_Y.cols()-1, 0, stop_Y.cols(), stop_Y.rows()), LaspMatrix<T>(1,1, iteration_costs[iteration_costs.size() - 2]));
				sv_stop_model.predict_confidence(last_sv, sv_mean, sv_sig);

				sv_stop_model.train(last_sv, last_Y, LaspMatrix<T>(1,1, iteration_costs.back()));
				
				if (*(r) > 50){
					r--;
					//*(r+1) = std::min(static_cast<int>(sv_stop_model.get_step()) + 1, p.n);
					//*(r+1) = std::min(static_cast<int>(sv_stop_model.get_step()) + 1, p.options.set_size);
					*(r+1) = std::min(static_cast<int>(sv_stop_model.get_cost_sensitive_step(static_cast<T>(p.n) + 1, error_thresh, error_prob)) + 1, p.options.set_size);
				}
				
				//cout << "EI per cost: " << sv_stop_model.get_step() << endl;
				//cout << "Lookahead: " << sv_stop_model.get_integer_step(static_cast<T>(p.n), error_thresh, error_prob) << endl;
				//cout << "Cost senstive step: " << *(r+1) << endl;
				
				T next_sv = static_cast<T>(*(r+1));
				sv_stop_model.predict_confidence(LaspMatrix<T>(1,1, next_sv), sv_mean, sv_sig);
				sv_stop_model.cost_predict_confidence(LaspMatrix<T>(1,1, next_sv), cost_mean, cost_sig);
				//cout << "Predicted score: " << sv_mean(0) << " sd: " << sv_sig(0) << endl;
				//cout << "Predicted cost: " << cost_mean(0) << " sd: " << cost_sig(0) << endl;
				
				LaspMatrix<T> target_error(1,1,current_error - error_thresh);
				//target_error.printMatrix("Target");
				target_error.normCDF(sv_mean, sv_sig);
				
				T target_error_prob = target_error(0);
				//cout << "Prob: " << target_error_prob << endl;
				
				if(*(r+1) >= p.options.set_size /*|| (*(r+1) > 100 && (target_error_prob < error_prob))*/) {
					cout << "Stopping step: " << *r << ", time: " << p.time.back() << endl;
					break;
				}
			}
			
			//if (!p.options.forward_stopping) {
//				if (stopvalue >= 0 && stopvalue < p.options.stoppingcriterion && S.size() > 10){
//					++stopIters;
//					
//					if (stopIters >= p.options.stopIters) {
//						break;
//					}
//				} else {
//					stopIters = 0;
//				}
			//}
			
			
		}
		
		p.y = y.template getRawArrayCopy<double>();
		p.xS = xS.template getRawArrayCopy<double>();
		p.S=S;
        
        //(Yu)
        //adding compressedSVM here
        /*
	if (p.options.compressedSVM) {
            
            cout << "Compressing..." << endl;
            
            // get the number of compressed SVs and randomly pick that much from the full SVs
            float ratio = p.options.compressedRatio;
            int totalSV = p.S.size();
            int numOfCompressedSVs = int(totalSV * ratio);
            
            if (p.options.verb > 1 ){
                cout << "\nTotal number of SV: " << totalSV << endl;
                cout << "Compressed Ratio: " << ratio << endl;
            }
            
            // make each support vector and its corresponding alpha together
            vector< pair<int, double> > SVIndice_alpha;
            for (int i = 0; i < p.S.size(); ++i) {
                // p.betas.back[0] is the bias(b) for the wx+b
                int SVIndice = p.S[i];
                double alpha = p.betas.back()[i+1];
                SVIndice_alpha.push_back( pair<int, double>(SVIndice, alpha) );
            }
            
            std::random_shuffle(SVIndice_alpha.begin(), SVIndice_alpha.end());
            
            // get the indices of selected SVs
            vector<int> selectedSVs;
            // get the corresponding y*alpha
            vector<double> selectedCoefs;
            
            vector< pair<int, double> >::iterator itr = SVIndice_alpha.begin();
            for (int i = 0; i < numOfCompressedSVs; ++i) {
                int SVIndice = (*itr).first;
                double alpha = (*itr).second;
                selectedSVs.push_back( SVIndice );
                selectedCoefs.push_back( alpha * p.y[SVIndice]);
                ++itr;
            }
            
            //test if randomly selection succeed
            //std::copy(selectedSVs.begin(), selectedSVs.end(), ostream_iterator<int>(cout, " "));
            //cout << endl;
            
            //std::copy(selectedCoefs.begin(), selectedCoefs.end(), ostream_iterator<double>(cout, " "));
            //cout << endl;
            
            // get the initial randomly picked SV matrix and selected coefficent matrix
            LaspMatrix<T> xSelectedSVs(0,0,0.0, selectedSVs.size(), p.features);
            x.gather(xSelectedSVs,selectedSVs);
            
            double* coefArray;
            double_vector_to_array(coefArray, selectedCoefs);
            LaspMatrix<T> xSelectedCoefs = LaspMatrix<double>(numOfCompressedSVs, 1, coefArray).convert<T>(); //1*numOfCompressedSVs
            
            
            LaspMatrix<T> params(1, numOfCompressedSVs * (p.features+1), 0.0);
            LaspMatrix<T> xSelectedSVsTranspose, xSelectedCoefsTranspose;
            xSelectedSVs.transpose(xSelectedSVsTranspose);
            xSelectedCoefs.transpose(xSelectedCoefsTranspose);
            
            int firstPart = numOfCompressedSVs * p.features;
            for (int i = 0; i < firstPart; ++i) {
                params(i) = xSelectedSVsTranspose(i);
            }
            for (int i = 0; i < numOfCompressedSVs; ++i) {
                params(firstPart + i) = xSelectedCoefsTranspose(i);
            }
            
            // conjugated gradient descent on selected SVs
            int maxiter = 1000;
            vector<T> losses;
            
            minimize(losses, maxiter, params, selectedSVs, x, y, totalSV, numOfCompressedSVs, p);
            
            //retrieve the data
            /*
             LaspMatrix<T> xNewSVsTranspose(p.features, selectedSVs.size(), 0.0);
             LaspMatrix<T> xNewCoefsTranspose(1, selectedSVs.size(), 0.0);
             LaspMatrix<T> xNewSVs(selectedSVs.size(), p.features, 0.0);
             LaspMatrix<T> xNewCoefs(selectedSVs.size(), 1, 0.0);
             
             for (int col = 0; col < p.features; ++col) {
             for (int row = 0; row < selectedSVs.size(); ++row) {
             xNewSVsTranspose(col,row) = params(col*selectedSVs.size() + row);
             }
             }
             
             for (int i = 0; i < numOfCompressedSVs; ++i) {
             xNewCoefsTranspose(i) = params(i + numOfCompressedSVs * p.features);
             }
             */
		/*
            p.xS = x.template getRawArrayCopy<double>();
            p.S.clear();
            p.S = selectedSVs;
            
        }
  */
		
		if(p.options.verb > 0){
			cout << endl << "Training Complete" << endl;
		}
		return CORRECT;
	}
  /*
    template<class T>
    void moveSVWeight(T &loss, LaspMatrix<T>& gradient, LaspMatrix<T>& params, vector<int>& selectedSVs, LaspMatrix<T>& x, LaspMatrix<T>& y, int numOfALLSVs, int numOfCompressedSVs, svm_problem& p){
        // x is d*n; y is 1*n; xSelectedSVs is d*nsv; xSelectedCoefs is 1*nsv(nsv is numOfCompressedSVs)
        
        LaspMatrix<T> xSelectedSVsTranspose(p.features, selectedSVs.size(), 0.0);
        LaspMatrix<T> xSelectedCoefsTranspose(1, selectedSVs.size(), 0.0);
        
        for (int col = 0; col < p.features; ++col) {
            for (int row = 0; row < selectedSVs.size(); ++row) {
                xSelectedSVsTranspose(col,row) = params(col*selectedSVs.size() + row);
            }
        }
        
        for (int i = 0; i < numOfCompressedSVs; ++i) {
            xSelectedCoefsTranspose(i) = params(i + numOfCompressedSVs * p.features);
        }
        
        LaspMatrix<T> xSelectedSVs, xSelectedCoefs;
        xSelectedSVsTranspose.transpose(xSelectedSVs);
        xSelectedCoefsTranspose.transpose(xSelectedCoefs);
        
        //xr(curse,:) = msv
        for (int i = 0; i < selectedSVs.size(); ++i) {
            x.setCol(selectedSVs[i], xSelectedSVs, i);
        }
        
        // compute the kernel
        // kernel options struct for computing the kernel
		kernel_opt kernelOptions;
		kernelOptions.kernel = p.options.kernel;
		kernelOptions.gamma = p.options.gamma;
		kernelOptions.degree = p.options.degree;
		kernelOptions.coef = p.options.coef;
        
        LaspMatrix<T> K;
        LaspMatrix<T> xNorm (p.n,1, 0.0);
        LaspMatrix<T> xSelectedSVsNorm(selectedSVs.size(), 1, 0.0);
        x.colSqSum(xNorm);
        xSelectedSVs.colSqSum(xSelectedSVsNorm);
        
        K.getKernel(kernelOptions, xSelectedSVs, xSelectedSVsNorm, x, xNorm); // K is nsv * p.n
        
        // compute the loss
        LaspMatrix<T> aOut,bOut;
        xSelectedCoefs.multiply(K, aOut); // aOut is 1*p.n
        aOut.subtract(y); // aOut is (alpha*K - preds)
        aOut.multiply(aOut, bOut, false, true); // this compute the l2-norm of aOut
        loss = 1.0/numOfALLSVs * bOut(0);
        
        // compute the gradient
        // compute the parta
        LaspMatrix<T> dsqrdparams;
        aOut.multiply(1.0/numOfALLSVs * 2.0, dsqrdparams); // dsqrdparams is 1*p.n
        LaspMatrix<T> svdsqrdparams;
        dsqrdparams.gather(svdsqrdparams, selectedSVs); // svdsqrdparams is 1*nsv
        
        LaspMatrix<T> KSelected;
        K.gather(KSelected,selectedSVs); //
        
        // parta = bsxfun(@times,Kr(cursv,:),alphay); % nsv*nsv
        LaspMatrix<T> parta_temp1;
        KSelected.rowWiseMult(xSelectedCoefs, parta_temp1); // parta_temp1 is nsv*nsv
        
        // parta = msv'*parta; % d*nsv
        LaspMatrix<T> parta_temp2, parta_temp3, parta_temp4, parta_temp5;
        xSelectedSVs.multiply(parta_temp1, parta_temp2, false, true); // parta_temp2 is d*nsv
        
        // parta = parta' - bsxfun(@times,Kr(cursv,:)'*alphay,msv); % nsv*d
        parta_temp2.transpose(parta_temp3); // parta_temp3 is nsv*d
        KSelected.multiply(xSelectedCoefs, parta_temp4, false, true); // Kr(cursv,:)'*alphay, parta_temp4 is nsv*1
        xSelectedSVsTranspose.colWiseMult(parta_temp4, parta_temp5); // parta5 is nsv*d
        parta_temp3.subtract(parta_temp5);
        
        // parta = bsxfun(@times,parta,svdsqrdparams);
        LaspMatrix<T> svdsqrdparamsTranspose, parta;
        svdsqrdparams.transpose(svdsqrdparamsTranspose);
        parta_temp3.colWiseMult(svdsqrdparamsTranspose, parta); // parta is nsv*d
        
        // compute the partb
        LaspMatrix<T> partb_temp1, partb_temp2, partb_temp3;
        // partb = bsxfun(@times,xr,dsqrdparams); % n*d
        x.rowWiseMult(dsqrdparams, partb_temp1); //partb_temp1 is d*n
        
        // partb = bsxfun(@times,Kr',alphay)*partb; % nsv*d
        K.colWiseMult(xSelectedCoefsTranspose, partb_temp2); //partb_temp2 is nsv*n
        partb_temp2.multiply(partb_temp1, partb_temp3, false, true); //partb_temp3 is nsv*d
        
        // partb = partb - bsxfun(@times,msv,(Kr'*dsqrdparams).*alphay); % nsv*d
        LaspMatrix<T> partb_temp4, partb_temp5, partb_temp6, partb;
        K.multiply(dsqrdparams, partb_temp4, false, true); //partb_temp4 is nsv*1
        partb_temp4.colWiseMult(xSelectedCoefsTranspose, partb_temp5); //partb_temp5 is nsv*1
        xSelectedSVsTranspose.colWiseMult(partb_temp5, partb_temp6); //partb_temp6 is nsv*d
        partb_temp3.subtract(partb_temp6, partb);
        
        // compute the partc
        // partc = (parta+partb) * 2 * sqrsig;
        LaspMatrix<T> partc;
        parta.add(partb, partc);
        partc.multiply(2.0 * kernelOptions.gamma); // partc is nsv*d
        
        LaspMatrix<T> gradientAlpha;
        // gradalpha = dsqrdparams'*Kr; % 1*nsv
        dsqrdparams.multiply(K, gradientAlpha, false, true); //gradientAlpha is 1*nsv
        
        // refresh the gradient
        int firstPart = numOfCompressedSVs * p.features;
        for (int i = 0; i< firstPart; ++i) {
            gradient(i) = partc(i);
        }
        int secondPart = firstPart + numOfCompressedSVs;
        for (int i = firstPart; i < secondPart; ++i) {
            gradient(i) = gradientAlpha(i - firstPart);
        }
    }
    
    template<class T>
    void minimize(vector<T>& fX, int length, LaspMatrix<T>& X, vector<int>& selectedSVs, LaspMatrix<T>& x, LaspMatrix<T>& y, int numOfALLSVs, int numOfCompressedSVs, svm_problem& p, T reduction){
        
        const T INTT = 0.1; //don't reevaluate within 0.1 of the limit of the current bracket
        const T EXTT = 3.0; //extrapolate maximum 3 times the current step-size
        const int MAXX = 20; //max 20 function evaluations per line search
        const T RATIO = 10.0; //maximum allowed slope ratio
        
        // SIG and RHO are the constants controlling the Wolfe-Powell conditions.
        // SIG is the maximum allowed absolute ratio between previous and new slopes (derivatives in the search direction),
        // thus setting SIG to low (positive) values forces higher precision in the line-searches.
        // RHO is the minimum allowed fraction of the expected (from the slope at the initial point in the linesearch).
        // Constants must satisfy 0 < RHO < SIG < 1.
        // Tuning of SIG (depending on the nature of the function to be optimized) may speed up the minimization;
        // it is probably not worth playing much with RHO.
        const T SIG = 0.1;
        T RHO = SIG / 2;
        
        // The "length" gives the length of the run: if it is positive, it gives the maximum number of line searches,
        // if negative its absolute gives the maximum allowed number of function evaluations.
        // The "reduction" indicates the reduction in function value to be expected in the first line-search (defaults to 1.0).
        string flag;
        length > 0 ? flag = "lineSearch" : flag = "functionEvaluation";
        
        int currentRun = 0; // initialize the counter of the run length
        bool lineSearchFailed = false; // no previous line search has failed by now
        
        T f0;
        LaspMatrix<T> df0(1, numOfCompressedSVs*p.features+numOfCompressedSVs, 0.0);
        moveSVWeight(f0, df0, X, selectedSVs, x, y, numOfALLSVs, numOfCompressedSVs, p);
        fX.push_back(f0);
        length > 0 ? currentRun += 1 : currentRun += 0;
        
        // initial search direction(steepest) and slope
        LaspMatrix<T> s;
        df0.multiply(-1, s);
        LaspMatrix<T> df0TimesS;
        df0.multiply(s, df0TimesS, true, false); // df0TimesS is 1*1
        
        T d0 = df0TimesS(0);
        // initial step is reduction/(|s|+1)
        T x3 = reduction / (1.0 - d0);
        
        while (currentRun < abs(length)) {
            // count iterations
            length > 0 ? currentRun += 1 : currentRun += 0;
            
            // make a copy of current values
            LaspMatrix<T> X0, dF0;
            X0.copy(X);
            dF0.copy(df0);
            T F0 ,M;
            F0 = f0;
            length > 0 ? M = MAXX : M = min(MAXX, -length - currentRun);
            
            T x1, f1=0.0, d1, x2, f2=0.0, d2, d3, f3=0.0, x4, f4=0.0, d4, A, B;
            LaspMatrix<T> df3, newX;
            //keep extrapolating as long as necessary
            while (true) {
                x2 = 0, f2 = f0, d2 = d0, f3 = f0;
                df3.copy(df0);
                bool success = false;
                
                while (!success && M > 0) {
                    try {
                        // count epochs
                        M = M -1;
                        length > 0 ? currentRun += 1 : currentRun += 0;
                        // newX = X+x3*s
                        s.multiply(x3, newX);
                        newX.add(X);
                        moveSVWeight(f3, df3, newX, selectedSVs, x, y, numOfALLSVs, numOfCompressedSVs, p);
                        if (isnan(f3) || isinf(f3)) {throw 1; }
                        for (int i = 0; i < df3.size(); ++i) {
                            if ( isnan(df3(i)) || isinf(df3(i)) ) throw 1;
                        }
                        success = true;
                    } catch (...) { //catch any error occured in moveSVWeight
                        //bisect and try again
                        x3 = (x2 + x3) / 2;
                    }
                }
                //keep best values
                if (f3 < F0) {X0.copy(newX); F0 = f3; dF0.copy(df3);}
                LaspMatrix<T> out;
                df3.multiply(s, out, true, false);
                //new slope
                d3 = out(0);
                // check if we are done extrapolating
                if (d3 > SIG * d0 || f3 > f0+x3*RHO*d0 || M == 0) break;
                // move point 2 to point 1
                x1 = x2; f1 = f2; d1 = d2;
                // move point 3 to point 2
                x2 = x3; f2 = f3; d2 = d3;
                // make cubic extrapolation
                A = 6*(f1-f2) + 3*(d2+d1)*(x2-x1);
                B = 3*(f2-f1) - (2*d1+d2)*(x2-x1);
                //num. error possible, ok!
                
                if (B*B-A*d1*(x2-x1) < 0) x3 = x2*EXTT; // same as isreal(x3)
                else x3 = x1 - d1*pow((x2-x1), 2)/( B + sqrt(B*B - A*d1*(x2-x1)) );
                
                if (isnan(x3) || isinf(x3) || x3<0) x3 = x2*EXTT;
                //if new point beyond extrapolation limit, extrapolate maximum amount
                else if (x3>x2*EXTT) x3 = x2*EXTT;
                //if new point is too close to previous point
                else if (x3 < x2+INTT*(x2-x1)) x3 = x2 + INTT*(x2-x1);
            }
            
            //keep interpolating
            while ((abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0) {
                //choose subinterval
                if (d3 > 0 || f3 > f0+x3*RHO*d0) {x4 = x3; f4 = f3; d4 =d3;} //move point 3 to point 4
                else {x2 = x3; f2 = f3; d2 = d3;} //move point 3 to point 2
                if (f4 > f0) {x3 = x2 - (0.5*d2*(x4-x2)*(x4-x2)) / (f4-f2-d2*(x4-x2));} //quadratic interpolation
                else {
                    A = 6*(f2-f4) / (x4-x2) + 3*(d4+d2); //cubic interpolation
                    B = 3*(f4-f2) - (2*d2+d4)*(x4-x2);
                    x3 = x2 + (sqrt(B*B-A*d2*(x4-x2)*(x4-x2))-B)/A; //num. error possible, ok!
                }
                if (isnan(x3) || isinf(x3)) x3 = (x2+x4) / 2; //if we had a numerical problem then bisect
                x3 = max( min(x3, x4-INTT*(x4-x2)), x2+INTT*(x4-x2)); //don't accept too close
                
                s.multiply(x3, newX);
                newX.add(X);
                moveSVWeight(f3, df3, newX, selectedSVs, x, y, numOfALLSVs, numOfCompressedSVs, p);
                
                //keep best values
                if (f3 < F0) { X0.copy(newX); F0 = f3; dF0.copy(df3); }
                // count epochs
                M = M -1;
                length > 0 ? currentRun += 1 : currentRun += 0;
                
                LaspMatrix<T> out;
                df3.multiply(s, out, true, false);
                //new slope
                d3 = out(0);
            }
            
            //if line search succeeded
            if (abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0) {
                //update variables
                s.multiply(x3, newX);
                newX.add(X);
                X.copy(newX);
                f0 = f3;
                fX.push_back(f0);
                
                if (p.options.verb > 1 ){
                cout << flag << " NO." << currentRun << "; Loss Value: " << f0 << endl;
                }
                else if (p.options.verb > 0) {
                    cout << ". " << flush;
                }
                
                LaspMatrix<T> temp1, temp2, temp3, temp4, temp5, temp6, temp7;
                
                df3.transpose(temp1);
                temp1.multiply(df3, temp2); //temp1 = df3'*df3
                df0.multiply(df3, temp3, true, false); //temp3 = df0'*df3
                
                df0.transpose(temp4);
                temp4.multiply(df0, temp5); //temp5 = df0'*df0
                s.multiply( (temp1(0)-temp3(0)) / temp5(0), temp6); //temp6 = (df3'*df3-df0'*df3)/(df0'*df0)*s
                temp6.subtract(df3, s); // s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;  Polack-Ribiere CG direction
                
                df0.copy(df3); //swap derivatives
                d3 = d0;
                df0.multiply(s, temp7, true, false); //temp7 = df0'*s
                d0 = temp7(0);
                //new slope must be negative, otherwise use steepest direction
                if (d0 > 0) {
                    df0.multiply(-1, s); // s = -df0;
                    LaspMatrix<T> temp8, temp9;
                    s.multiply(-1, temp8);
                    temp8.multiply(s, temp9, true, false);
                    d0 = temp9(0);
                }
                x3 = x3 * min(RATIO, d3/(d0-std::numeric_limits<T>::min())); //slope ratio but max RATIO
                lineSearchFailed = false; //this line search did not fail
            }
            else{
                //restore best point so far
                X.copy(X0); f0 = F0; df0.copy(dF0);
                //line search failed twice in a row or we ran out of time, so we give up
                if (lineSearchFailed || currentRun > abs(length)) break;
                
                df0.multiply(-1, s);
                LaspMatrix<T> temp10;
                df0.multiply(s,temp10, true, false); //temp10 = -s'*s;
                //try steepest
                d0 = temp10(0);
                
                x3 = 1.0 / (1-d0);
                lineSearchFailed = true;
            }
        }
    }
  */	
	void tempOutputCheck(svm_node** x, double* y){
		ofstream out;
		out.open("outputTest.txt");
		for(int i = 0; i < 3; i++){
			out << "Node " << i / 3 << ": " << x[i / 3][i % 3].index << " " << x[i / 3][i % 3].value << ", Y: " << y[i/3] << endl;
		}
		out.close();
	}
	 
	
	template int lasp_svm_host<float>(svm_problem& p);
	template int lasp_svm_host<double>(svm_problem& p);
    
}


void lasp::setup_svm_problem_shuffled(svm_problem& problem,
									  svm_sparse_data myData,
									  svm_sparse_data& holdoutData,
									  opt options,
									  int posClass,
									  int negClass)
{
	if(myData.allData.size() > 2) {
		if (problem.options.verb > 0) {
			cerr << "myData must only have 2 classes!" << endl;
		}
		exit_with_help();
	}
	
	srand (time(0));
	
	problem.options = options;
	problem.features = myData.numFeatures;
	//a vector of support vectors and their associated classification
	//in sparse form.
	vector<pair<vector<svm_node>, double> > allData;
	
	int numDataPoints = 0;
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator myIter = myData.allData.begin(); myIter != myData.allData.end(); ++myIter) {
        
		numDataPoints += myIter->second.size();
		for(int dataPoint = 0; dataPoint < myIter->second.size(); ++dataPoint) {
			pair<vector<svm_node>, double> curVector;
			curVector.first = myIter->second[dataPoint];
			curVector.second = myIter->first;
			allData.push_back(curVector);
		}
	}
	problem.classifications.push_back(posClass);
	problem.classifications.push_back(negClass);
	
	
	//now, we need to shuffle allData
	if (problem.options.shuffle){
		random_shuffle(allData.begin(), allData.end());
	}
	//here, we pop off 30% of the data as "holdout data" to be used to
	//accomplish platt scaling later.
	vector<pair<vector<svm_node>, double> > holdout;
	for(int i = 0; i < .3 * allData.size(); ++i) {
		holdout.push_back(allData.back());
		allData.pop_back();
	}
	
	holdoutData.orderSeen.push_back(posClass); holdoutData.orderSeen.push_back(negClass);
	holdoutData.numFeatures = myData.numFeatures;
	holdoutData.numPoints = holdout.size();
	holdoutData.multiClass = false;
	//now lets fill up the holdoutData.
	for(int i = 0; i < holdout.size(); ++i) {
		pair<vector<svm_node>, double> curPair = holdout[i];
		holdoutData.allData[int(curPair.second)].push_back(curPair.first);
	}
	
	problem.n = numDataPoints - holdout.size();
	
	//now that we've shuffled everything and pulled out the holdout data,
	//lets put all the data into vectors.
	vector<double> fullDataVector;
	//classificationVector must contain -1 and 1 only, as surrogates for the
	//classes. The lesser of the two classes is represented by -1 and the greater
	//of the two classes is 1.
	vector<double> classificationVector;
	
	for(int x = 0; x < problem.n; ++x) {
		vector<double> fullDataPoint;
		sparse_vector_to_full(fullDataPoint, allData[x].first, myData.numFeatures);
		for(int i = 0; i < myData.numFeatures; ++i) fullDataVector.push_back(fullDataPoint[i]);
		if(allData[x].second == negClass)
			classificationVector.push_back(-1);
		else //it must be the positive class
			classificationVector.push_back(1);
	}
	double_vector_to_array(problem.xS, fullDataVector);
	double_vector_to_array(problem.y, classificationVector);
    
    problem.means = myData.means;
    problem.standardDeviations = myData.standardDeviations;
    
    
    featureScaling<double>(problem.xS, problem.features, problem.n, problem.means, problem.standardDeviations);
}


void lasp::setup_svm_problem_shuffled(svm_problem& problem,
									  svm_sparse_data myData,
									  opt options,
									  int posClass,
									  int negClass)
{
    if(myData.allData.size() > 2) {
		cout << "myData must only have 2 classes!" << endl;
		exit_with_help();
	}
	
	srand (time(0));
	
	problem.options = options;
	problem.features = myData.numFeatures;
	//a vector of support vectors and their associated classification
	//in sparse form.
	vector<pair<vector<svm_node>, double> > allData;
	
	int numDataPoints = 0;
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator myIter = myData.allData.begin();
		myIter != myData.allData.end();
		++myIter) {
		numDataPoints += myIter->second.size();
		for(int dataPoint = 0; dataPoint < myIter->second.size(); ++dataPoint) {
			//			pair<vector<svm_node>, double> curVector(;
			//			curVector.first = ;
			//			curVector.second = ;
			allData.push_back(make_pair(myIter->second[dataPoint], myIter->first));
		}
	}
	problem.classifications.push_back(posClass);
	problem.classifications.push_back(negClass);
	
	problem.n = numDataPoints;
	
	//now, we need to shuffle allData
	if(problem.options.shuffle){
		random_shuffle(allData.begin(), allData.end());
	}
	
	//now that we've shuffled everything, lets put all the data into vectors.
	vector<double> fullDataVector;
	//classificationVector must contain -1 and 1 only, as surrogates for the
	//classes. The lesser of the two classes is represented by -1 and the greater
	//of the two classes is 1.
	vector<double> classificationVector;
	
	for(int x = 0; x < problem.n; ++x) {
		vector<double> fullDataPoint;
		sparse_vector_to_full(fullDataPoint, allData[x].first, myData.numFeatures);
		for(int i = 0; i < myData.numFeatures; ++i) fullDataVector.push_back(fullDataPoint[i]);
		if(allData[x].second == negClass)
			classificationVector.push_back(-1);
		else //it must be the positive class
			classificationVector.push_back(1);
	}
	
	double_vector_to_array(problem.xS, fullDataVector);
	double_vector_to_array(problem.y, classificationVector);
    
    //(Yu)
    
    //problem.means = myData.means;
    //problem.standardDeviations = myData.standardDeviations;
    
    //featureScaling<double>(problem.xS, problem.features, problem.n, problem.means, problem.standardDeviations);
}

//(Yu)
template<class T>
void lasp::featureScaling(T* data, int numFeatures, int numPoints, vector<double>& means, vector<double>& standardDeviations){
    
    for (int i = 0; i < numFeatures; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            data[i+j*numFeatures] -= means[i];
            data[i+j*numFeatures] /= standardDeviations[i];
        }
    }
}

template void lasp::featureScaling<double>(double* data, int numFeatures, int numPoints, vector<double>& means, vector<double>& standardDeviations);

template void lasp::featureScaling<float>(float* data, int numFeatures, int numPoints, vector<double>& means, vector<double>& standardDeviations);




