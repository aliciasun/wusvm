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
#include "predict.h"
#include "kernels.h"
#include "fileIO.h"
#include <math.h>
#include <list>



void lasp::classify_host(svm_model& myModel,
						 svm_sparse_data& sparseData,
						 int& correct,
						 char* outputfile)
{
	lasp::svm_full_data testData;
	sparse_data_to_full(testData, sparseData);
    
    //(Yu) normalize the test data
//    featureScaling<float>(testData.x, testData.numFeatures, testData.numPoints, myModel.means, myModel.standardDeviations);
//    
	map<int, vector<int> > classificationVectors;
	for(int i = 0; i < myModel.orderSeen.size(); ++i) {
		for(int j = i+1; j < myModel.orderSeen.size(); ++j) {
			lasp::svm_binary_model curModel;
			int c1, c2;
			c1 = myModel.orderSeen[i]; c2 = myModel.orderSeen[j];
			curModel = get_binary_model(myModel, c1, c2);
			
			double* classifications = new double[sparseData.numPoints];
			lasp::classify_host(classifications, curModel, testData, myModel.options);
			for(int k = 0; k < sparseData.numPoints; ++k) {
				int guess = classifications[k] > 0 ? c1 : c2;
				classificationVectors[k].push_back(guess);
			}
			
			delete[] classifications;
		}
	}
	
	//popular vote implementation
	vector<int> finalClassifications;
	for(int i = 0; i < testData.numPoints; ++i) {
		map<int, int> voteCounts;
		for(int j = 0; j < myModel.orderSeen.size(); ++j) {
			voteCounts[myModel.orderSeen[j]] = 0;
		}
		
		for(int j = 0; j < classificationVectors[i].size(); ++j) {
			voteCounts[classificationVectors[i][j]] ++;
		}
		
		int popCount = -1;
		int popClass = myModel.orderSeen[0];
		for(map<int,int>::iterator iter = voteCounts.begin();
			iter != voteCounts.end();
			++iter) {
			if(iter->second > popCount) {
				popCount = iter->second;
				popClass = iter->first;
			}
		}
		
        //(Yu) the order of finalClassification is not the orginal order of test points
		finalClassifications.push_back(popClass);
	}
    
    
	
	correct = 0;
	for(int i = 0; i < finalClassifications.size(); ++i) {
		if(finalClassifications[i] == testData.y[i]) ++correct;
	}
	
	//This is literally the worst thing of all time (Gabe)
	int unclassified = -99999, index = 0;
	vector<int> outputClassifications(finalClassifications.size(), unclassified);
	
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator iter = sparseData.allData.begin(); iter != sparseData.allData.end(); ++iter) {
		int curClass = iter->first;
		int outputIndex = 0;
		
		for (; testData.y[index] == curClass; ++index) {
			for (; outputIndex < sparseData.pointOrder.size(); ++outputIndex) {
				if (sparseData.pointOrder[outputIndex] == curClass) {
					outputClassifications[outputIndex] = finalClassifications[index];
					++outputIndex;
					break;
				}
			}
		}
	}
	
	if (outputfile != 0) {
		output_classifications(outputfile, outputClassifications);
	}
	//(Yu) the order of outputClassifications is the orginal order of test points
	sparseData.outputClassifications = outputClassifications;
	
}



void lasp::dagclassify_host(svm_model& myModel,
                            svm_sparse_data& sparseData,
                            int& correct,
                            char* outputfile) {
	
	lasp::svm_full_data testData;
	sparse_data_to_full(testData, sparseData);
    
    //(Yu) normalize the test data
    featureScaling<float>(testData.x, testData.numFeatures, testData.numPoints, myModel.means, myModel.standardDeviations);
    
    
	int numClasses = myModel.orderSeen.size();
	int numPoints = sparseData.numPoints;
	
	// Each vector in this list contains the indexes for one partition
	// of the data. Size of list increases by one for each iteration of
	// outer loop, reaching numClasses by the end.
	list<vector<int> > indexes;
	vector<int> allIndexes;
	for (int i = 0; i < numPoints; ++i) {
		allIndexes.push_back(i);
	}
	
	indexes.push_back(allIndexes);
	// Each iteration of outer loop can be seen as one level of the DAG
    
	for(int i = 1; i < numClasses; ++i) {
		indexes.push_back(vector<int>());
		// Each iteration of inner loop can be seen as one node on level i
		// of the DAG.
		
		for(int j = 0; j < i; ++j) {
			indexes.push_back(vector<int>());
			lasp::svm_binary_model curModel;
			int c1, c2;
			c1 = myModel.orderSeen[j]; c2 = myModel.orderSeen[numClasses - i + j];
			curModel = get_binary_model(myModel, c1, c2);
			
			// Create arrays to hold the data and copy it over. No way around this
			// that I could think of.
			float *x = new float[testData.numFeatures * indexes.front().size()];
			float *y = new float[indexes.front().size()];
			for (int k = 0; k < indexes.front().size(); ++k) {
				int start = indexes.front()[k] * testData.numFeatures;
				int end = (indexes.front()[k] + 1) * testData.numFeatures;
				y[k] = 0.0;
				std::copy(testData.x + start, testData.x + end, x + testData.numFeatures * k);
			}
			
			// Create the full data object
			lasp::svm_full_data data;
			data.numFeatures = testData.numFeatures;
			data.numPoints = indexes.front().size();
			data.x = x;
			data.y = y;
			
			// Classify and update index vectors.
			double classifications[data.numPoints];
			lasp::classify_host(classifications, curModel, data, myModel.options);
			for (int k = 0; k < data.numPoints; ++k) {
				if (classifications[k] > 0) {
					(++indexes.rbegin())->push_back(indexes.front()[k]);
				} else {
					indexes.back().push_back(indexes.front()[k]);
				}
			}
			
			// Done with this node.
			indexes.pop_front();
		}
	}
    
	// Label instances based on membership in the index vectors.
	vector<int> finalClassifications(testData.numPoints);
	int i = 0;
	for (std::list<vector<int> >::iterator it = indexes.begin(); it != indexes.end(); ++it,++i) {
		vector<int> vec = *it;
		for (int j = 0; j < vec.size(); ++j) {
			finalClassifications[vec[j]] = myModel.orderSeen[i];
		}
	}
	
	correct = 0;
	for(int i = 0; i < finalClassifications.size(); ++i) {
		if(finalClassifications[i] == testData.y[i]) ++correct;
	}
    
	int unclassified = -99999, index = 0;
	vector<int> outputClassifications(finalClassifications.size(), unclassified);
    
	//Reorder points to match the input order
	typedef map<int, vector<vector<svm_node> > >::iterator SparseIterator;
	for(SparseIterator iter = sparseData.allData.begin(); iter != sparseData.allData.end(); ++iter) {
		int curClass = iter->first;
		int outputIndex = 0;
		
		for (; index < finalClassifications.size() && testData.y[index] == curClass; ++index) {
			for (; outputIndex < sparseData.pointOrder.size(); ++outputIndex) {
				if (sparseData.pointOrder[outputIndex] == curClass) {
					outputClassifications[outputIndex] = finalClassifications[index];
					++outputIndex;
					break;
				}
			}
		}
	}
    
	if (outputfile != 0) {
		output_classifications(outputfile, outputClassifications);
	}
	
	sparseData.outputClassifications = outputClassifications;
}



lasp::svm_binary_model lasp::get_binary_model(lasp::svm_model& sparseModel,
											  int positiveClass,
											  int negativeClass)
{
	svm_binary_model returnModel;
	returnModel.kernelType = sparseModel.kernelType;
	returnModel.numFeatures = sparseModel.numFeatures;
	returnModel.degree = sparseModel.degree;
	returnModel.coef = sparseModel.coef;
	returnModel.gamma = sparseModel.gamma;
	returnModel.pegasos = sparseModel.pegasos;
	returnModel.b = sparseModel.offsets[positiveClass][negativeClass];
	
	map<vector<svm_node>, map<int, double>, CompareSparseVectors >& posSV = sparseModel.modelData[positiveClass];
	map<vector<svm_node>, map<int, double>, CompareSparseVectors >& negSV = sparseModel.modelData[negativeClass];
	
	vector<double> betas;
	vector<vector<double> > fullSupportVectors;
	
	typedef map<vector<svm_node>, map<int, double> >::iterator SVIterator;
	for(SVIterator iter = posSV.begin();
		iter != posSV.end();
		++iter) {
		vector<svm_node> curSparse = iter->first;
		vector<double> curFull;
		int curSparseIndex = 0;
		for(int i = 0; i < sparseModel.numFeatures; ++i) {
			if(curSparse.size() > 0 && curSparseIndex < curSparse.size() && curSparse[curSparseIndex].index == i+1) {
				curFull.push_back(curSparse[curSparseIndex].value);
				curSparseIndex++;
			}
			else {
				curFull.push_back(0);
			}
		}
		
		fullSupportVectors.push_back(curFull);
		betas.push_back(iter->second[negativeClass]);
	}
	
	for(SVIterator iter = negSV.begin();
		iter != negSV.end();
		++iter) {
		vector<svm_node> curSparse = iter->first;
		vector<double> curFull;
		int curSparseIndex = 0;
		for(int i = 0; i < sparseModel.numFeatures; ++i) {
			if(curSparse.size() > 0 && curSparseIndex < curSparse.size() && curSparse[curSparseIndex].index == i+1) {
				curFull.push_back(curSparse[curSparseIndex].value);
				curSparseIndex++;
			}
			else {
				curFull.push_back(0);
			}
		}
		
		fullSupportVectors.push_back(curFull);
		betas.push_back(iter->second[positiveClass]);
	}
	
	returnModel.numSupportVectors = fullSupportVectors.size();
	double_vector_to_float_array_two_dim(returnModel.xS, fullSupportVectors);
	double_vector_to_float_array(returnModel.betas, betas);
	
	return returnModel;
}

int lasp::classify_host(double* classifications,
						lasp::svm_binary_model& myModel,
						lasp::svm_full_data& myData, opt& options,
						bool distFromHyperplane)
{
	//Set GPU parameters
	if (options.usegpu && DeviceContext::instance()->getNumDevices() < 1) {
		if (options.verb > 0) {
			cerr << "No CUDA device found, reverting to CPU-only version" << endl;
		}
		
		options.usegpu = false;
	} else if (options.maxGPUs > -1){
		DeviceContext::instance()->setNumDevices(options.maxGPUs);
	}
	
	//Get a matrix of the weights to multiply in later
	LaspMatrix<float> betas(myModel.numSupportVectors, 1);
	
#pragma omp parallel for
	for (int i = 0; i < betas.cols(); ++i) {
		betas(i, 0) = myModel.betas[i];
	}
	
	//Create data matricies
	LaspMatrix<float> dXS(myModel.numSupportVectors, myModel.numFeatures, myModel.xS);
	LaspMatrix<float> dXe(myData.numPoints, myData.numFeatures, myData.x);
    
	//Check for implicit bias used in pegasos training
	bool pegasos = myModel.pegasos;
	double bias = pegasos ? 0 : myModel.b;
	
	//Account for potential difference in number of realized features in train/test data
	int totalFeatures = 0;
	if (pegasos) {
        
		totalFeatures = std::max(dXS.rows(), dXe.rows() + 1);
		if (totalFeatures > dXS.rows()) {
			LaspMatrix<float> lastRowS = dXS(0, dXS.rows()-1, dXS.cols(), dXS.rows()).copy();
			dXS.resize(dXS.cols(), dXS.rows()-1);
			dXS.resize(dXS.cols(), totalFeatures, true, true, 0.0);
			dXS.setRow(dXS.rows()-1, lastRowS);
		}
		
		dXe.resize(dXe.cols(), totalFeatures, true, true, 0.0);
		LaspMatrix<float> lastRow = dXe(0, dXe.rows()-1, dXe.cols(), dXe.rows());
		lastRow.add(1.0);
	} else {
        
		totalFeatures = std::max(dXS.rows(), dXe.rows());
		dXS.resize(dXS.cols(), totalFeatures, true, true, 0.0);
		dXe.resize(dXe.cols(), totalFeatures, true, true, 0.0);
	}
	
	// Update the features and data in the model to reflect changes
	myModel.numFeatures = totalFeatures;
	myModel.xS = dXS.data();
	myModel.pegasos = false; //Update should not be done more than once
	myData.numFeatures = totalFeatures;
	myData.x = dXe.data();
	
	LaspMatrix<float> dXe_copy, dXS_copy;
    
	// Get copies of data matricies if we're copying them to the device
	if(options.usegpu){
		dXe_copy.transferToDevice();
		dXS_copy.transferToDevice();
		betas.transferToDevice();
		
		int error = dXe_copy.copy(dXe);
		error += dXS_copy.copy(dXS);
		
		if (error != 0) {
			dXe_copy = dXe;
			dXS_copy = dXS;
		}
	} else {
		dXe_copy = dXe;
		dXS_copy = dXS;
	}
	
	//Calculate norms
	LaspMatrix<float> dXeNorm(myData.numPoints,1);
	dXe_copy.colSqSum(dXeNorm);
	
	LaspMatrix<float> dXSNorm(myData.numPoints,1);
	dXS_copy.colSqSum(dXSNorm);
	
	//Set kernel options
	kernel_opt kernelOptions;
	kernelOptions.kernel = myModel.kernelType;
	kernelOptions.gamma = myModel.gamma;
	kernelOptions.degree = myModel.degree;
	kernelOptions.coef = myModel.coef;
	
	//Calculate chunks to break up test points into if we're using the small kernel option
	int maxElem = 0;
	int numChunks = 0;
	int chunkStart = 0;
	
	//Make sure our kernel allocation succeeds
	LaspMatrix<float> Ke;
	while(true){
		maxElem = options.set_size;
		numChunks = options.smallKernel ? 1 + ((myData.numPoints - 1) / maxElem) : 1;
		
		//Kernel matrix
		try {
			Ke.resize(myData.numPoints / numChunks, myModel.numSupportVectors);
			break;
		} catch (std::bad_alloc) {
			options.smallKernel = true;
			options.set_size /= 2;
		}
	}
	
	if (options.usegpu) {
		Ke.transferToDevice();
	}
	
	//Loop through the chunks
	for (int chunk = 0; chunk < numChunks; ++chunk) {
		int chunkSize = (chunk == numChunks - 1) ? myData.numPoints - chunkStart : myData.numPoints / numChunks;
		int chunkEnd = chunkStart + chunkSize;
		
		if (chunkSize == 0) {
			continue;
		}
		
		//Get chunk kernel matrix
		LaspMatrix<float> dXe_chunk = dXe_copy(chunkStart, 0, chunkEnd, dXe.rows());
		LaspMatrix<float> dXeNorm_chunk = dXeNorm(chunkStart, 0, chunkEnd, 1);
		Ke.getKernel(kernelOptions, dXS_copy, dXSNorm, dXe_chunk, dXeNorm_chunk, false, false, options.usegpu);
		LaspMatrix<float> classes;
		
		betas.multiply(Ke, classes);
		classes.add(bias);
		classes.transferToHost();  //Avoid race condition in operator()
		
		//Set classes
		if(!distFromHyperplane){
#pragma omp parallel for
			for (int i=0; i < chunkSize; ++i) {
				classifications[i + chunkStart] = classes(i, 0) > 0 ? 1 : -1;
			}
		}
		else {
#pragma omp parallel for
			for (int i=0; i < chunkSize; ++i) {
				classifications[i + chunkStart] = classes(i, 0);
			}
		}
		
		chunkStart = chunkEnd;
	}
	
	
	dXS.release();
	dXe.release();
	return 0;
}



