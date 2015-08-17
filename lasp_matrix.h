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

#ifndef LASP_MATRIX_H
#define LASP_MATRIX_H

#include "abstract_matrix.h"
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <ostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string.h>
#include <limits>
#include <memory>

#include "blas_wrappers.h"

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lasp
{
	
	
	template<class T>
	struct EvalBase;
	
	template<class T>
    class LaspMatrix {
		
		//Use pointers so data can be transferred effficiently
		// rc: reference count, key: memory identification key
		// Offset/end used for submatricies
		size_t *cols_, *rows_, *mCols_, *mRows_,  *colOffset_, *rowOffset_, *colEnd_, *rowEnd_;
		int *rc_, *subrc_;
		void **key_;
		bool *device_, *registered_, *unified_;
		DeviceContext *context_;
		
		//The data itself, the format of which vary based on what particular
		//subclass we're in.
		T **data_, **dData_;
		
		//Private accessors given by reference
		inline int& _rc() const{ return *rc_; };
		inline int& _subrc() const{ return *subrc_;};
		inline void*& _key() const{ return *key_; };
		inline size_t& _cols() const{ return *cols_; };
		inline size_t& _mCols() const{ return *mCols_; };
		inline size_t& _rows() const{ return *rows_; };
		inline size_t& _mRows() const{ return *mRows_; };
		inline size_t& _colOffset() const{ return *colOffset_; }
		inline size_t& _rowOffset() const{ return *rowOffset_; }
		inline size_t& _colEnd() const{ return *colEnd_; }
		inline size_t& _rowEnd() const{ return *rowEnd_; }
		inline bool& _device() const{ return *device_; }
		inline bool& _registered() const{ return *registered_; }
		inline bool& _unified() const{ return *unified_; }
		inline T*& _data() const{ return *data_; };
		inline T*& _dData() const{ return *dData_; }
		
		//Internal methods for reference counting management
		void cleanup();
		void freeData();
		
		void laspAlloc(size_t size);
		void laspFree(T* ptr);
		
#ifdef CUDA
		cudaError_t laspCudaAlloc(void** ptr, size_t size);
#endif
		
		//Internal device methods
		int deviceCopy(LaspMatrix<T> &other);
		LaspMatrix<T> deviceCopy();
		
		int deviceResize(size_t newCols, size_t newRows, bool copy = true, bool fill = false, T val = 0.0);
		
		int deviceSetRow(size_t row, LaspMatrix<T>& other);
		int deviceSetCol(size_t col, LaspMatrix<T>& other);
		
		int deviceSetRow(size_t row, LaspMatrix<T>& other, size_t otherRow);
		int deviceSetCol(size_t col, LaspMatrix<T>& other, size_t otherCol);
		
	public:
		//Public accessors for member variables
		inline int rc() const{ return *rc_; };
		inline int subrc() const{ return *subrc_;};
		inline void* key() const{ return *key_; };
		inline size_t cols() const{ return (*colEnd_ != 0) ? max(min(*cols_, *colEnd_) - *colOffset_, (size_t)0) : max(*cols_ - *colOffset_, (size_t)0); };
		inline size_t mCols() const{ return *mCols_ - *colOffset_; };
		inline size_t rows() const{ return (*rowEnd_ != 0) ? max(min(*rows_, *rowEnd_) - *rowOffset_, (size_t)0) : max(*rows_ - *rowOffset_, (size_t)0); };
		inline size_t mRows() const{ return *mRows_; };
		inline size_t colOffset() const{ return *colOffset_; }
		inline size_t rowOffset() const{ return *rowOffset_; }
		inline size_t colEnd() const{ return *colEnd_; }
		inline size_t rowEnd() const{ return *rowEnd_; }
		inline size_t size() const{ return (size_t)rows() * (size_t)cols(); };
		inline size_t mSize() const{ return (size_t)(*mRows_) * (size_t)(*mCols_) - ((size_t)(*colOffset_) * (size_t)(*mRows_) + (size_t)(*rowOffset_)); };
		inline size_t elements() const{ return size(); };
		inline size_t mElements() const{ return size(); };
		inline T* data() const{ return *data_ + *mRows_ * *colOffset_ + *rowOffset_; };
		inline T* dData() const{ return *dData_ + *mRows_ * *colOffset_ + *rowOffset_; }
		inline bool device() const{ return *device_ || *unified_; }
		inline bool registered() const{ return *registered_; }
		inline bool unified() const{ return *unified_; }
		inline bool isSubMatrix() const { return rowOffset() != 0 || colOffset() != 0 || rowEnd() != 0 || colEnd() != 0; };
		inline DeviceContext& context() const { return *context_; }
		
		
		//CONSTRUCTORS: Here, constructors only allocate
		//memory, rather than setting it.
		
		//Default Constructor
		LaspMatrix();
		

		//NOTE: mcol and mrow are used to pre-allocate additional space for a matrix
		//this is useful when you know a matrix, for instance the set of basis vectors
		//used for training the svm, starts out small, but will need to grow later

		//Standard Constructor, d is pointer to pre-allocated data
		LaspMatrix(size_t col, size_t row, T* d = 0, size_t mCol = 0, size_t mRow = 0);
		
		//Fill Constructor, val is the initial value for all elements
		LaspMatrix(size_t col, size_t row, T val, size_t mCol = 0, size_t mRow = 0, bool fill=true, bool fill_mem=false);
		
		//Construct from c++ vector
		LaspMatrix(vector<T> vec);
		
		//Copy Constructor: Same as assignment operator
		LaspMatrix(const LaspMatrix<T> &other);
		
		//Copy data into new memory
		int copy(LaspMatrix<T> &other, bool copyMem=false);
		LaspMatrix<T> copy(bool copyMem=false);
		
		
		//Resize matrix
		int resize(size_t newCols, size_t newRows, bool copy = true, bool fill = false, T val = 0.0, bool no_swap=false);
		
		//Release data (NOT SAFE, USE WITH CAUTION!!!)
		int release();
		
		//DESTRUCTOR
		~LaspMatrix();
		
		//Transfer data to/from device
		int transfer();
		//Safer ways to call transfer
		int transferToDevice();
		int transferToHost();
		int registerHost();
		
		//Assignment Operator: Assigns to same data in memory, managed with reference counting
		LaspMatrix<T>& operator=(const LaspMatrix<T>& other);
		LaspMatrix<T>& operator=(const T& val);
		
		//EvalBase conversion
		template<class N>
		LaspMatrix(const shared_ptr<N >& other);
		
		template<class N>
		LaspMatrix<T>& operator=(const shared_ptr<N >& other);
		
		//Element access operator
		T& operator()(int index);
		T& operator()(int col, int row);
		LaspMatrix<T> operator()(size_t startCol, size_t startRow, size_t endCol, size_t endRow);
		
		//Raw memory access
		T& operator[](size_t matrixPosition);
		
		//Copy
		int setRow(size_t row, LaspMatrix<T>& other);
		int setCol(size_t col, LaspMatrix<T>& other);
		
		int setRow(size_t row, LaspMatrix<T>& other, size_t otherRow);
		int setCol(size_t col, LaspMatrix<T>& other, size_t otherCol);
		
		//Multiplication with BLAS-like syntax
		int multiply(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix, bool transposeMe = false, bool transposeOther = false, T a = 1.0, T b = 0.0, int numRowsToSkip = 0);
		int multiply(LaspMatrix<T>& otherMatrix, bool transposeMe = false, bool transposeOther = false, T a = 1.0, T b = 0.0, int numRowsToSkip = 0);
		int multiply(T scalar, LaspMatrix<T>& outputMatrix);
		int multiply(T scalar);
		
		int transpose(LaspMatrix<T>& outputMatrix);
		int transpose();
		
		LaspMatrix<T> diag(bool column = true);
		int diagAdd(T scalar);
		
		T maxElem(int& col, int& row);
		T maxElem();
		
		T minElem(int& col, int& row);
		T minElem();
		
		int add(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix);
		int add(LaspMatrix<T>& otherMatrix);
		int add(T scalar, LaspMatrix<T>& outputMatrix);
		int add(T scalar);
		
		int subtract(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix);
		int subtract(LaspMatrix<T>& otherMatrix);
		int subtract(T scalar, LaspMatrix<T>& outputMatrix);
		int subtract(T scalar);
		
		int negate(LaspMatrix<T>& output);
		int negate();
		
		int colWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output);
		int colWiseMult(LaspMatrix<T>& vec);
		
		int rowWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output);
		int rowWiseMult(LaspMatrix<T>& vec);
		
		int pow(T exp, LaspMatrix<T>& output);
		int pow(T exp);
		
		int exp(LaspMatrix<T>& output, T gamma = -1);
		int exp(T gamma = -1);
		
		int tanh(LaspMatrix<T>& output);
		int tanh();
		
		int log(LaspMatrix<T>& output);
		int log();
		
		int normCDF(LaspMatrix<T>& output, T mean = 0, T sd = 1.0);
		int normCDF(T mean = 0, T sd = 1.0);
		int normCDF(LaspMatrix<T>& output, LaspMatrix<T>& mean, LaspMatrix<T>& sd);
		int normCDF(LaspMatrix<T>& mean, LaspMatrix<T>& sd);
		
		int normPDF(LaspMatrix<T>& output, T mean = 0, T sd = 1.0);
		int normPDF(T mean = 0, T sd = 1.0);
		int normPDF(LaspMatrix<T>& output, LaspMatrix<T>& mean, LaspMatrix<T>& sd);
		int normPDF(LaspMatrix<T>& mean, LaspMatrix<T>& sd);
		
		int solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, LaspMatrix<T>& LU, LaspMatrix<int>& ipiv);
		int solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output);
		int solve(LaspMatrix<T>& otherMatrix);
		
		int chol(LaspMatrix<T>& output);
		int chol();
		
		int cholSolve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output);
		int cholSolve(LaspMatrix<T>& otherMatrix);
		
		int colSqSum(LaspMatrix<T>& output, T scalar = 1);
		LaspMatrix<T> colSqSum(T scalar = 1);
		
		int colSum(LaspMatrix<T>& output, T scalar = 1);
		LaspMatrix<T> colSum(T scalar = 1);
		
		int eWiseOp(LaspMatrix<T>& output, T add, T mult, T pow1);
		LaspMatrix<T> eWiseOp(T add, T mult, T pow1);
		
		int eWiseDivM( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1 = 1, T pow2 = 1);
		LaspMatrix<T> eWiseDivM( LaspMatrix<T>& otherMatrix, T pow1 = 1, T pow2 = 1);
		
		int eWiseMultM( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1 = 1, T pow2 = 1);
		LaspMatrix<T> eWiseMultM( LaspMatrix<T>& otherMatrix, T pow1 = 1, T pow2 = 1);
		
		int eWiseScale( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, int d0, int d);
		LaspMatrix<T> eWiseScale(LaspMatrix<T>& otherMatrix, int d0, int d);
		
		T trace();
		
		//Just swaps rows and columns
		int swap();
		
		template<class ITER>
		int gather(LaspMatrix<T>& output, ITER begin, ITER end);
		int gather(LaspMatrix<T>& output, LaspMatrix<int> map);
		int gather(LaspMatrix<T>& output, vector<int>& map);
		
		template<class ITER>
		LaspMatrix<T> gather(ITER begin, ITER end);
		LaspMatrix<T> gather(LaspMatrix<int> map);
		LaspMatrix<T> gather(vector<int>& map);
		
		template<class ITER>
		int gatherSum(LaspMatrix<T>& output, ITER begin, ITER end);
		int gatherSum(LaspMatrix<T>& output, LaspMatrix<int> map);
		int gatherSum(LaspMatrix<T>& output, vector<int>& map);
		
		int contigify(LaspMatrix<int>& map);
		int revert(LaspMatrix<int>& map);
		
		int shuffle();
		int shuffle(LaspMatrix<int>& revert_map);
		int shuffle(LaspMatrix<int> &revert_map, LaspMatrix<int> &shuffle_map);
		
		LaspMatrix<T> gatherMap(vector<int>& map);
		
		LaspMatrix<T> getSubMatrix(size_t startCol, size_t startRow, size_t endCol, size_t endRow);
		LaspMatrix<T> getSubMatrix(size_t endCol, size_t endRow);
		
		int addMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out);
		int subMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out);
		
		int add_outer(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix);
		int add_outer(LaspMatrix<T>& otherMatrix);
		
		int ger(LaspMatrix<T>& output, LaspMatrix<T> X, LaspMatrix<T> Y, T alpha = 1);
		int ger(LaspMatrix<T> X, LaspMatrix<T> Y, T alpha = 1);
		
		int getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& Xnorm1, LaspMatrix<T>& X2, LaspMatrix<T>& Xnorm2, LaspMatrix<T>& l, bool mult = false, bool transMult = false, bool useGPU = true);
		int getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& Xnorm1, LaspMatrix<T>& X2, LaspMatrix<T>& Xnorm2, bool mult = false, bool transMult = false, bool useGPU = true);
		int getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& X2, LaspMatrix<T>& l, bool mult = false, bool transMult = false, bool useGPU = true);
		int getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& X2, bool mult = false, bool transMult = false, bool useGPU = true);
		
		template<class N>
		LaspMatrix<N> convert(bool mem=false);
		
		void printMatrix(string name = "", int c = 0, int r = 0);
		void printInfo(string name = "");
		void checksum(string name = "");
		
		template<class N>
		N* getRawArrayCopy();
		
		bool operator==(LaspMatrix<T>& other);
		
		//Static special matrix generators
		static LaspMatrix<T> eye(size_t n);
		static LaspMatrix<T> zeros(size_t cols, size_t rows);
		static LaspMatrix<T> zeros(size_t n);
		static LaspMatrix<T> ones(size_t cols, size_t rows);
		static LaspMatrix<T> ones(size_t n);
		static LaspMatrix<T> random(size_t cols, size_t rows);
		static LaspMatrix<T> random(size_t n);
		static LaspMatrix<T> vcat(LaspMatrix<T> top, LaspMatrix<T> bottom);
		static LaspMatrix<T> hcat(LaspMatrix<T> left, LaspMatrix<T> right);
	};
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(): rc_(new int), rows_(new size_t), cols_(new size_t), mCols_(new size_t), mRows_(new size_t), colOffset_(new size_t), rowOffset_(new size_t), colEnd_(new size_t), rowEnd_(new size_t), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), unified_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = 0;
		_cols() = 0;
		_mRows() = 0;
		_mCols() = 0;
		_data() = 0;
		_dData() = 0;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_device() = false;
		_registered() = false;
		_unified() = context().getUseUnified();
		_key() = context().getNextKey();
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(size_t col, size_t row, T* d, size_t mCol, size_t mRow): rc_(new int), rows_(new size_t), cols_(new size_t), mCols_(new size_t), mRows_(new size_t), colOffset_(new size_t), rowOffset_(new size_t), colEnd_(new size_t), rowEnd_(new size_t), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), unified_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = row;
		_cols() = col;
		_mRows() = mRow == 0 ? row : mRow;
		_mCols() = mCol == 0 ? col : mCol;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_dData() = 0;
		_device() = false;
		_registered() = false;
		_unified() = context().getUseUnified();
		_key() = context().getNextKey();
		
		_data() = d;
		
		if (d == 0) {
			laspAlloc((size_t)_mCols() * (size_t)_mRows());
		} else if(unified()) {
			*this = this->copy();
		}
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(size_t col, size_t row, T val, size_t mCol, size_t mRow, bool fill, bool fill_mem): rc_(new int), rows_(new size_t), cols_(new size_t), mCols_(new size_t), mRows_(new size_t), colOffset_(new size_t), rowOffset_(new size_t), colEnd_(new size_t), rowEnd_(new size_t), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), unified_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = row;
		_cols() = col;
		_mRows() = mRow == 0 ? row : mRow;
		_mCols() = mCol == 0 ? col : mCol;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_dData() = 0;
		_key() = context().getNextKey();
		_device() = false;
		_registered() = false;
		_unified() = context().getUseUnified();
		laspAlloc(_mRows() * _mCols());
		
		if(fill && !fill_mem){
			size_t rowsTemp = rows();
			size_t colsTemp = cols();
			size_t mRowsTemp = mRows();
			T* dataTemp = data();
			
			for(size_t i = 0; i < colsTemp; ++i){
				for(size_t j = 0; j < rowsTemp; ++j){
					dataTemp[i * mRowsTemp + j] = val;
				}
			}
			
		} else if(fill_mem){
			fill_n(data(), mSize(), val);
		}
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(vector<T> vec): rc_(new int), rows_(new size_t), cols_(new size_t), mCols_(new size_t), mRows_(new size_t), colOffset_(new size_t), rowOffset_(new size_t), colEnd_(new size_t), rowEnd_(new size_t), subrc_(new int), data_(new T*), dData_(new T*), device_(new bool), registered_(new bool), unified_(new bool), context_(DeviceContext::instance()), key_(new void*){
		_rc() = 1;
		_subrc() = 1;
		_rows() = 0;
		_cols() = 0;
		_mRows() = 0;
		_mCols() = 0;
		_data() = 0;
		_dData() = 0;
		_rowOffset() = 0;
		_colOffset() = 0;
		_rowEnd() = 0;
		_colEnd() = 0;
		_device() = false;
		_registered() = false;
		_unified() = context().getUseUnified();
		_key() = context().getNextKey();
		
		resize(vec.size(), 1);
		for (int i = 0; i < vec.size(); ++i) {
			operator()(i) = vec[i];
		}
	}
	
	template<class T>
	LaspMatrix<T>::LaspMatrix(const LaspMatrix<T>& other): rc_(other.rc_), rows_(other.rows_), cols_(other.cols_), mCols_(other.mCols_), mRows_(other.mRows_), colOffset_(other.colOffset_), rowOffset_(other.rowOffset_), colEnd_(other.colEnd_), rowEnd_(other.rowEnd_), subrc_(other.subrc_),  data_(other.data_), dData_(other.dData_), context_(other.context_), device_(other.device_), registered_(other.registered_), unified_(other.unified_), key_(other.key_){
		_rc()++;
		_subrc()++;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eye(size_t n){
		LaspMatrix<T> retVal(n, n, 0.0);
		for(size_t i = 0; i < n; ++i){
			retVal.operator()(i,i) = 1.0;
		}
		
		return retVal;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::zeros(size_t cols, size_t rows){
		return LaspMatrix<T>(cols, rows, 0.0);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::zeros(size_t n){
		return LaspMatrix<T>::zeros(n, n);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::ones(size_t cols, size_t rows){
		return LaspMatrix<T>(cols, rows, 1.0);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::ones(size_t n){
		return LaspMatrix<T>::ones(n, n);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::random(size_t cols, size_t rows){
		LaspMatrix<T> retVal(cols, rows);
		size_t mrowsTemp = rows;
		T* dataTemp = retVal.data();
		
		T max = 1.0, min = 0.0;
		for(size_t j = 0; j < cols; ++j){
			for (size_t i = 0; i < rows; ++i){
				dataTemp[mrowsTemp * j + i] = static_cast <T> ((rand()) / (static_cast <T> (RAND_MAX/(max - min)) + min));
			}
		}
		
		return retVal;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::random(size_t n){
		return LaspMatrix<T>::random(n, n);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::vcat(LaspMatrix<T> top, LaspMatrix<T> bottom){
		if (top.size() == 0) {
			return bottom.copy();
		}
		
		if (bottom.size() == 0){
			return top.copy();
		}
		
		LaspMatrix<T> output(top.cols(), top.rows() + bottom.rows());
		output(0, 0, output.cols(), top.rows()).copy(top);
		output(0, top.rows(), output.cols(), output.rows()).copy(bottom);
		return output;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::hcat(LaspMatrix<T> left, LaspMatrix<T> right){
		if (left.size() == 0) {
			return right.copy();
		}
		
		if (right.size() == 0){
			return left.copy();
		}
		
		LaspMatrix<T> output(left.cols() + right.cols(), left.rows());
		output(0, 0, left.cols(), output.rows()).copy(left);
		output(left.cols(), 0, output.cols(), output.rows()).copy(right);
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::copy(LaspMatrix<T>& other, bool copyMem){
		int resizeResult = resize(other.cols(), other.rows());
		
		if(resizeResult != MATRIX_SUCCESS){
			return resizeResult;
		}
		
		if(device() || other.device()){
			return deviceCopy(other);
		}
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		T* other_dataTemp = other.data();
		size_t other_mrowsTemp = other.mRows();
		
#ifdef _OPENMP
		size_t ompCount = colsTemp * rowsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				dataTemp[mrowsTemp * j + i] = other_dataTemp[other_mrowsTemp * j + i];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::copy(bool copyMem){
		LaspMatrix<T> result;
		result.copy(*this, copyMem);
		return result;
	}
	
	template<class T>
	int LaspMatrix<T>::release(){
#ifndef NDEBUG
		if(rc() > 1){
			cerr << "Warning: Releasing data with multiple references" << endl;
		}
#endif
		_rc() = -1;
		_subrc() = -1;
		cleanup();
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	void LaspMatrix<T>::cleanup(){
		if (--_rc() == 0 && key_ != 0) {
			freeData();
			delete rc_;
			if(dData_ != data_) delete dData_;
			delete data_;
			delete rows_;
			delete cols_;
			delete mRows_;
			delete mCols_;
			delete key_;
			delete device_;
			delete registered_;
			delete unified_;
			key_ = 0;
		}
		
		if(--_subrc() == 0 && key_ != 0){
			delete subrc_;
			delete colOffset_;
			delete rowOffset_;
			delete colEnd_;
			delete rowEnd_;
		}
	}
	
	template<class T>
	LaspMatrix<T>::~LaspMatrix<T>(){
		if(key_ != 0){
			cleanup();
		}
	}
	
	template<class T>
	LaspMatrix<T>& LaspMatrix<T>::operator=(const LaspMatrix<T>& other){
		cleanup();
		
		rc_ = other.rc_;
		subrc_ = other.subrc_;
		_rc()++;
		_subrc()++;
		
		cols_ = other.cols_;
		rows_ = other.rows_;
		mCols_ = other.mCols_;
		mRows_ = other.mRows_;
		data_ = other.data_;
		key_ = other.key_;
		dData_ = other.dData_;
		device_ = other.device_;
		registered_ = other.registered_;
		unified_ = other.unified_;
		context_ = other.context_;
		rowOffset_ = other.rowOffset_;
		colOffset_ = other.colOffset_;
		rowEnd_ = other.rowEnd_;
		colEnd_ = other.colEnd_;
		
		
		return *this;
	}
	
	template<class T>
	LaspMatrix<T>& LaspMatrix<T>::operator=(const T& val){
		resize(1, 1);
		this->operator()(0) = val;
		return *this;
	}
	
	template<class T>
	T& LaspMatrix<T>::operator()(int index) {
		if(device()){
			transferToHost();
		}
		
#ifndef NDEBUG
		if(index >= rows() * cols()){
			cerr << "Error: Index (" << index << ") out of bounds!" << endl;
			throw INVALID_DIMENSIONS;
		}
#endif
		
		size_t row = index % rows();
		size_t col = index / rows();
		
		return data()[col * (size_t)mRows() + row];
	}
	
	template<class T>
	T& LaspMatrix<T>::operator()(int col, int row) {
		if(device()){
			transferToHost();
		}
		
#ifndef NDEBUG
		if(col >= cols() || row >= rows()){
			cerr << "Error: Index (" << col << ", " << row << ") out of bounds!" << endl;
			throw INVALID_DIMENSIONS;
		}
#endif
		return data()[col * (size_t)mRows() + row];
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::operator()(size_t startCol, size_t startRow, size_t endCol, size_t endRow){
		return getSubMatrix(startCol, startRow, endCol, endRow);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::getSubMatrix(size_t startCol, size_t startRow, size_t endCol, size_t endRow){
#ifndef NDEBUG
		if(startCol >= cols() || startRow >= rows() || startRow > endRow || startCol > endCol){
			cerr << "Error: Indicies (" << startCol << ", " << startRow << ") :: (" << endCol << ", " << endRow << ") out of bounds!" << endl;
		}
#endif
		
		LaspMatrix<T> output(*this);
		
		output.colOffset_ = new size_t;
		output.rowOffset_ = new size_t;
		output.colEnd_ = new size_t;
		output.rowEnd_ = new size_t;
		output.subrc_ = new int;
		
		--_subrc();
		output._subrc() = 1;
		output._colOffset() = startCol + colOffset();
		output._rowOffset() = startRow + rowOffset();
		output._colEnd() = endCol == 0 ? endCol : colOffset() + endCol;
		output._rowEnd() = endRow == 0 ? endRow : rowOffset() + endRow;
		
		return output;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::getSubMatrix(size_t endCol, size_t endRow){
		return getSubMatrix(0, 0, endCol, endRow);
	}
	
	template<class T>
	T& LaspMatrix<T>::operator[](size_t matrixPosition){
#ifndef NDEBUG
		//cerr << "Warning: LaspMatrix::operator[] will result in unchecked memory access, consider using LaspMatrix::operator() instead" << endl;
#endif
		return data()[matrixPosition];
	}
	
	template<class T>
	int LaspMatrix<T>::setRow(size_t row, LaspMatrix<T>& other){
		if(other.size() != cols()){
			cerr << "Error: Dimension mismatch in setRow" << endl;
			return INVALID_DIMENSIONS;
		}
		
		if(device() || other.device()){
			return deviceSetRow(row, other);
		}
		
		for(int i = 0; i < other.size(); ++i){
			operator()(i, row) = other(i);
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::setRow(size_t row, LaspMatrix<T>& other, size_t otherRow){
		if(other.cols() != cols()){
			cerr << "Error: Dimension mismatch in setRow" << endl;
			return INVALID_DIMENSIONS;
		}
		
		if(device() || other.device()){
			return deviceSetRow(row, other, otherRow);
		}
		
		for(size_t i = 0; i < other.cols(); ++i){
			operator()(i, row) = other(i, otherRow);
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::setCol(size_t col, LaspMatrix<T>& other){
		if(other.size() != rows()){
			cerr << "Error: Dimension mismatch in setCol" << endl;
			return INVALID_DIMENSIONS;
		}
		
		if(device() || other.device()){
			return deviceSetCol(col, other);
		}
		
		for(size_t i = 0; i < other.size(); ++i){
			operator()(col, i) = other(i);
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::setCol(size_t col, LaspMatrix<T>& other, size_t otherCol){
		if(other.rows() != rows()){
			cerr << "Error: Dimension mismatch in setCol" << endl;
			return INVALID_DIMENSIONS;
		}
		
		if(device() || other.device()){
			return deviceSetCol(col, other, otherCol);
		}
		
		size_t rowsTemp = rows();
		T* dataTemp = &(operator()(col, 0));
		T* otherDataTemp = &(other(otherCol, 0));
		
		#ifdef _OPENMP
		size_t ompCount = rowsTemp;
		size_t ompLimit = context().getOmpLimit();
		#endif
		
		#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t i = 0; i < rowsTemp; ++i){
			dataTemp[i] = otherDataTemp[i];
		}
		
		return MATRIX_SUCCESS;
	}
	
	//TODO: Check excess fill values
	template<class T>
	int LaspMatrix<T>::resize(size_t newCols, size_t newRows, bool copy, bool fill, T val, bool no_swap){
		if(cols() == newCols && rows() == newRows){
			return MATRIX_SUCCESS;
		}
		
		if(isSubMatrix()){
#ifndef NDEBUG
			cerr << "Cannot resize a sub-matrix" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		
		if(device()){
			return deviceResize(newCols, newRows, copy, fill, val);
		}
		
		if (((newCols == 1 && _rows() == 1 && _cols() != 1) || (newRows == 1 && _cols() == 1 && _rows() != 1)) && !no_swap){
			std::swap(_rows(), _cols());
			std::swap(_mRows(), _mCols());
		}
		
		if(_mRows() < newRows || _mCols() < newCols){
			if ((max(newCols, _cols()) * max(newRows, _rows()) * 8) / 100000000.0 > 1.0){
#ifndef NDEBUG
				cerr << "Allocating size: " << (max(newCols, _cols()) * max(newRows, _rows()) * 8) / 1000000000.0 << " GB" << endl;
#endif
			}
			
			T* oldptr = _data();
			laspAlloc((size_t)max(newCols, cols()) * (size_t)max(newRows, rows()));
			T* newptr = _data();
			
			
			if(copy && rows() > 0 && cols() > 0){
#ifdef _OPENMP
				size_t ompCount = _rows() * _cols();
				size_t ompLimit = context().getOmpLimit();
#endif
				
#pragma omp parallel for if(ompCount > ompLimit)
				for(size_t i = 0; i < _rows(); ++i){
					for(size_t j = 0; j < _cols(); ++j){
						//Fix this to use memcpy
						newptr[j * (size_t)newRows + i] = oldptr[j * (size_t)_rows() + i];
					}
				}
			}
			
			if(fill){
				//Done stupidly on purpose, fix at some point
#ifdef _OPENMP
				size_t ompCount = newRows * newCols;
				size_t ompLimit = context().getOmpLimit();
#endif
				
#pragma omp parallel for if(ompCount > ompLimit)
				for(size_t i = 0; i < newRows; ++i){
					for(size_t j = 0; j < newCols; ++j){
						if(i >= _rows() || j >= _cols()){
							newptr[j * (size_t)newRows + i] = val;
						}
					}
				}
				
			}
			
			laspFree(oldptr);
			_mCols() = newCols;
			_mRows() = newRows;
		}
		
		_cols() = newCols;
		_rows() = newRows;
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::multiply( LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix, bool transposeMe, bool transposeOther, T a, T b, int numRowsToSkip){
		return METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::multiply(LaspMatrix<T>& otherMatrix, bool transposeMe, bool transposeOther, T a, T b, int numRowsToSkip){
		return this->multiply(otherMatrix, *this, transposeMe, transposeOther, a, b);
	}
	
	template<class T>
	int LaspMatrix<T>::multiply(T scalar, LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, 0, scalar, 1);
		}
		
		outputMatrix.transferToHost();
		
		//Resize output
		bool copy = outputMatrix.key() == key();
		outputMatrix.resize(cols(), rows(), copy);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] * scalar;
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::multiply(T scalar) {
		return multiply(scalar, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::transpose(LaspMatrix<T> &outputMatrix){
		if (device()) {
			int error = MATRIX_SUCCESS;
			error += outputMatrix.transferToDevice();
			error += outputMatrix.resize(rows(), cols());
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &outputMatrix);
				return device_transpose(params, dData(), outputMatrix.dData(), cols(), rows(), mRows(), outputMatrix.mRows());
			}
		}
		
		transferToHost();
		outputMatrix.transferToHost();
		outputMatrix.resize(rows(), cols());
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * i + j] = dataTemp[mrowsTemp * j + i];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::transpose(){
		if (isSubMatrix()) {
#ifndef NDEBUG
			cerr << "Cannot transpose subMatrix!" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		if(mRows() == 1 || mCols() == 1){
			return this->swap();
		}
		
		LaspMatrix<T> output;
		transpose(output);
		
		std::swap(_data(), output._data());
		std::swap(_dData(), output._dData());
		std::swap(_cols(), output._cols());
		std::swap(_rows(), output._rows());
		std::swap(_mCols(), output._mCols());
		std::swap(_mRows(), output._mRows());
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::diag(bool column){
		_mRows()++;
		LaspMatrix<T> subMat = this->operator()(0, 0, min(cols(), rows()), 1);
		LaspMatrix<T> output;
		subMat.transpose(output);
		_mRows()--;
		
		if (!column){
			output.transpose();
		}
		
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::diagAdd(T scalar){
		_mRows()++;
		LaspMatrix<T> subMat = this->operator()(0, 0, min(cols(), rows()), 1);
		int error = subMat.add(scalar);
		_mRows()--;
		
		return error;
	}
	
	template<class T>
	T LaspMatrix<T>::maxElem(int& col, int& row) {
		transferToHost();
		
		col = 0;
		row = 0;
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		T maxE = std::numeric_limits<T>::min();
		
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				T elem = dataTemp[mrowsTemp * j + i];
				if(elem > maxE){
					maxE = elem;
					col = j;
					row = i;
				}
			}
		}
		
		return maxE;
	}
	
	template<class T>
	T LaspMatrix<T>::maxElem() {
		int col, row;
		return maxElem(col, row);
	}
	
	template<class T>
	T LaspMatrix<T>::minElem(int& col, int& row) {
		transferToHost();
		
		col = 0;
		row = 0;
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		T minE = std::numeric_limits<T>::max();
		
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				T elem = dataTemp[mrowsTemp * j + i];
				if(elem < minE){
					minE = elem;
					col = j;
					row = i;
				}
			}
		}
		
		return minE;
	}
	
	template<class T>
	T LaspMatrix<T>::minElem() {
		int col, row;
		return minElem(col, row);
	}
	
	template<class T>
	int LaspMatrix<T>::add(LaspMatrix<T> &otherMatrix, LaspMatrix<T> &outputMatrix) {
		if (size() == 1) {
			return otherMatrix.add(operator()(0), outputMatrix);
		}
		
		if (otherMatrix.size() == 1) {
			return add(otherMatrix(0), outputMatrix);
		}
		
		if (cols() != otherMatrix.cols() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in add" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		outputMatrix.resize(cols(), rows(), copy);
		
		
		if (device()){
			int error = MATRIX_SUCCESS;
			
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &outputMatrix);
				return device_addMatrix(params,dData(),otherMatrix.dData(),outputMatrix.dData(), rows(), cols(), mRows(), otherMatrix.mRows(), outputMatrix.mRows());
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();
		
		T* other_dataTemp = otherMatrix.data();
		size_t other_mrowsTemp = otherMatrix.mRows();
		
#ifdef _OPENMP
		size_t ompCount = colsTemp * rowsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] + other_dataTemp[other_mrowsTemp * j + i];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::add(LaspMatrix<T> &otherMatrix) {
		return add(otherMatrix, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::add(T scalar, LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, scalar, 1, 1);
		}
		
		outputMatrix.transferToHost();
		
		//Resize output
		bool copy = outputMatrix.key() == key();
		outputMatrix.resize(cols(), rows(), copy);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();
		
#ifdef _OPENMP
		size_t ompCount = colsTemp * rowsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] + scalar;
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::add(T scalar) {
		return add(scalar, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::eWiseMultM(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1, T pow2){
		if(otherMatrix.rows() != rows() || otherMatrix.cols() != cols()){
#ifndef NDEBUG
			cerr << "Error: Matrix dimensions must be equal for eWiseDivM" << endl;
#endif
			return INVALID_DIMENSIONS;
		}
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows(), false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &output);
				device_eWiseMult(params, dData(), otherMatrix.dData(), output.dData(), size(), pow1, pow2, rows(), mRows(), otherMatrix.mRows(), output.mRows());
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
		T* other_dataTemp = otherMatrix.data();
		size_t other_mrowsTemp = otherMatrix.mRows();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for( size_t j = 0; j < colsTemp; ++j){
			for( size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[j * output_mrowsTemp + i] = std::pow(dataTemp[j*mrowsTemp + i], pow1) * std::pow(other_dataTemp[j * other_mrowsTemp + i], pow2);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseMultM(LaspMatrix<T>& otherMatrix, T pow1, T pow2){
		LaspMatrix<T> output;
		int err = eWiseMultM(otherMatrix, output, pow1, pow2);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(LaspMatrix<T> &otherMatrix, LaspMatrix<T> &outputMatrix) {
		if (size() == 1) {
			int retval = otherMatrix.subtract(operator()(0), outputMatrix);
			retval += outputMatrix.negate();
			return retval;
		}
		
		if (otherMatrix.size() == 1) {
			return subtract(otherMatrix(0), outputMatrix);
		}
		
		
		if (cols() != otherMatrix.cols() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in subtract" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		outputMatrix.resize(cols(), rows(), copy);
		
		if (device()){
			int error = MATRIX_SUCCESS;
			
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &outputMatrix);
				return device_subMatrix(params,dData(),otherMatrix.dData(),outputMatrix.dData(), rows(), cols(), mRows(), otherMatrix.mRows(), outputMatrix.mRows());
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = outputMatrix.mRows();
		T* output_dataTemp = outputMatrix.data();
		
		T* other_dataTemp = otherMatrix.data();
		size_t other_mrowsTemp = otherMatrix.mRows();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[mrowsTemp * j + i] - other_dataTemp[other_mrowsTemp * j + i];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(LaspMatrix<T> &otherMatrix) {
		return subtract(otherMatrix, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(T scalar, LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, -scalar, 1, 1);
		}
		
		return add(-scalar, outputMatrix);
	}
	
	template<class T>
	int LaspMatrix<T>::subtract(T scalar) {
		return subtract(scalar, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::negate(LaspMatrix<T> &outputMatrix) {
		if(device()){
			return eWiseOp(outputMatrix, 0, -1, 1);
		}
		
		return multiply(-1.0, outputMatrix);
	}
	
	template<class T>
	int LaspMatrix<T>::negate() {
		return negate(*this);
	}
	
	template<class T>
	int LaspMatrix<T>::colWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output){
		// mat is x by y, vec has length x
		if (!((vec.rows() == rows() && vec.cols() == 1) || (vec.cols() == rows() && vec.rows() == 1))){
			cerr << "Error: you must pass a vector with the same number of rows as the input matrix" << endl;
			return INVALID_DIMENSIONS;
		}
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += vec.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows());
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &vec, &output);
				device_colWiseMult(params, dData(), output.dData(), vec.dData(), rows(), cols(), mRows(), output.mRows(), (vec.rows() == 1 ? vec.mRows() : 1));
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		vec.transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t colsTemp = cols();
		size_t rowsTemp = rows();
		size_t mRowsTemp = mRows();
		
		size_t output_mrowsTemp = output.mRows();
		size_t vec_stride = (vec.rows() == rows() && vec.cols() == 1) ? 1 : vec.mRows();
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		T* vecData = vec.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for (size_t i = 0; i < colsTemp; ++i){
			for (size_t j =0; j < rowsTemp; ++j){
				output_dataTemp[output_mrowsTemp * i + j] = dataTemp[mRowsTemp * i + j] * vecData[vec_stride * j];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::colWiseMult(LaspMatrix<T>& vec){
		return colWiseMult(vec, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::rowWiseMult(LaspMatrix<T>& vec, LaspMatrix<T>& output){
		// mat is x by y, vec has length x
		if (!((vec.rows() == cols() && vec.cols() == 1) || (vec.cols() == cols() && vec.rows() == 1))){
			cerr << "Error: you must pass a vector with the same number of rows as the input matrix" << endl;
			return INVALID_DIMENSIONS;
		}
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += vec.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows());
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &vec, &output);
				device_rowWiseMult(params, dData(), output.dData(), vec.dData(), rows(), cols(), mRows(), output.mRows(), (vec.rows() == 1 ? vec.mRows() : 1));
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		vec.transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t colsTemp = cols();
		size_t rowsTemp = rows();
		size_t mRowsTemp = mRows();
		
		size_t output_mrowsTemp = output.mRows();
		size_t vec_stride = (vec.rows() == rows() && vec.cols() == 1) ? 1 : vec.mRows();
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		T* vecData = vec.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for (size_t j =0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output[output_mrowsTemp * j + i]= dataTemp[mRowsTemp * j + i] * vecData[vec_stride * j];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::rowWiseMult(LaspMatrix<T>& vec){
		return rowWiseMult(vec, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::pow(T exp, LaspMatrix<T>& output){
		if(device()){
			return eWiseOp(output, 0, 1, exp);
		}
		
		output.transferToHost();
		
		//Resize output
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = std::pow(dataTemp[mrowsTemp * j + i], exp);
			}
		}
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::pow(T exp){
		return pow(exp, *this);
	}
	
	template<class T>
	int LaspMatrix<T>::exp(LaspMatrix<T>& output, T gamma){
		//Resize output
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_exp(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows(), gamma);
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
		size_t sizeTemp = rowsTemp * colsTemp;
		
#ifdef _OPENMP
		size_t ompCount = sizeTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t k = 0; k < sizeTemp; ++k){
			size_t i = k % rowsTemp;
			size_t j = k / rowsTemp;
			output_dataTemp[output_mrowsTemp * j + i] = std::exp(dataTemp[mrowsTemp * j + i] * -gamma);
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::exp(T gamma){
		return exp(*this, gamma);
	}
	
	
	template<class T>
	int LaspMatrix<T>::tanh(LaspMatrix<T>& output){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_tanh(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows());
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = std::tanh(dataTemp[mrowsTemp * j + i]);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::tanh(){
		return tanh(*this);
	}
	
	
	template<class T>
	int LaspMatrix<T>::log(LaspMatrix<T>& output){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_log(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows());
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = std::log(dataTemp[mrowsTemp * j + i]);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::log(){
		return log(*this);
	}
	
	template<class T>
	int LaspMatrix<T>::normCDF(LaspMatrix<T>& output, T mean, T sd){
		//Normalize if needed
		if (mean != 0 || sd != 1) {
			int error = subtract(mean, output);
			error += output.multiply(1.0 / sd);
			
			if (error != MATRIX_SUCCESS){
				return error;
			}
			
			return output.normCDF();
		}
		
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_normCDF(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows());
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = cdf(dataTemp[mrowsTemp * j + i]);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::normCDF(T mean, T sd){
		return normCDF(*this, mean, sd);
	}
	
	template<class T>
	int LaspMatrix<T>::normCDF(LaspMatrix<T>& output, LaspMatrix<T>& mean, LaspMatrix<T>& sd){
		if (mean.size() == 1 && sd.size() == 1) {
			return normCDF(output, mean(0), sd(0));
		}
		
		int error = subtract(mean, output);
		error += output.eWiseDivM(sd, output);
		
		if (error != MATRIX_SUCCESS){
			return error;
		}
		
		return output.normCDF();
	}
	
	template<class T>
	int LaspMatrix<T>::normCDF(LaspMatrix<T>& mean, LaspMatrix<T>& sd){
		return normCDF(*this, mean, sd);
	}
	
	template<class T>
	int LaspMatrix<T>::normPDF(LaspMatrix<T>& output, T mean, T sd){
		//Normalize if needed
		if (mean != 0 || sd != 1) {
			int error = subtract(mean, output);
			error += output.multiply(1.0 / sd);
			error += output.normPDF(output);
			
			if (error != MATRIX_SUCCESS){
				return error;
			}
			
			return output.multiply(1.0 / sd);
		}
		
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_normPDF(params,dData(), output.dData(), cols(), rows(), mRows(), output.mRows());
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = pdf(dataTemp[mrowsTemp * j + i]);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::normPDF(T mean, T sd){
		return normPDF(*this, mean, sd);
	}
	
	template<class T>
	int LaspMatrix<T>::normPDF(LaspMatrix<T>& output, LaspMatrix<T>& mean, LaspMatrix<T>& sd){
		if (mean.size() == 1 && sd.size() == 1) {
			return normPDF(output, mean(0), sd(0));
		}
		
		int error = subtract(mean, output);
		error += output.eWiseDivM(sd, output);
		error += output.normPDF();
		
		if (error != MATRIX_SUCCESS){
			return error;
		}
		
		return output.eWiseDivM(sd, output);
	}
	
	template<class T>
	int LaspMatrix<T>::normPDF(LaspMatrix<T>& mean, LaspMatrix<T>& sd){
		return normPDF(*this, mean, sd);
	}
	
	template<class T>
	int LaspMatrix<T>::colSqSum(LaspMatrix<T>& output, T scalar){
		
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(), 1);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_colSqSum(params, dData(), cols(), rows(), output.dData(), scalar, mRows(), output.mRows());
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), 1);
		T* newptr = output.data();//new T[cols()];
		
		memset(newptr, 0, cols()*sizeof(T));
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t i = 0; i < colsTemp; ++i){
			for (size_t j = 0; j < rowsTemp; ++j){
				T val = dataTemp[ i * mrowsTemp + j];
				newptr[i] += val * val;
			}
			newptr[i] *= scalar;
		}
		//output = LaspMatrix<T>(cols(), 1, newptr);
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::colSqSum(T scalar){
		LaspMatrix<T> output;
		int err = colsSqSum(output, scalar);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::colSum(LaspMatrix<T>& output, T scalar){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(), 1, false, true, 0.0);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_colSum(params, dData(), cols(), rows(), output.dData(), scalar, mRows(), output.mRows());
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		output.resize(cols(), 1);
		
		T* newptr = output.data();//new T[cols()];
		
		memset(newptr, 0, cols()*sizeof(T));
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t i = 0; i < colsTemp; ++i){
			for (size_t j = 0; j < rowsTemp; ++j){
				T val = dataTemp[ i * mrowsTemp + j];
				newptr[i] += val;
			}
			newptr[i] *= scalar;
		}
		//output = LaspMatrix<T>(cols(), 1, newptr);
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::colSum(T scalar){
		LaspMatrix<T> output;
		int err = colSum(output, scalar);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::eWiseOp(LaspMatrix<T> &output, T add, T mult, T pow1){
		if (device()){
			int error = output.transferToDevice();
			error += output.resize(cols(),rows(),false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_ewiseOp(params, dData(), output.dData(), size(), add, mult, pow1, rows(), mRows(), output.mRows());
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
		if(pow1 == 1 || mult == 0){
#pragma omp parallel for if(ompCount > ompLimit)
			for (size_t j = 0; j < colsTemp; ++j){
				for ( size_t i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  dataTemp[j * mrowsTemp + i] * mult + add;
				}
			}
		} else if(pow1 == 2.0){
#pragma omp parallel for if(ompCount > ompLimit)
			for (size_t j = 0; j < colsTemp; ++j){
				for ( size_t i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  dataTemp[j * mrowsTemp + i] * dataTemp[j * mrowsTemp + i] * mult + add;
				}
			}
		} else if(pow1 == 0.5){
#pragma omp parallel for if(ompCount > ompLimit)
			for (size_t j = 0; j < colsTemp; ++j){
				for ( size_t i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  sqrt(dataTemp[j * mrowsTemp + i]) * mult + add;
				}
			}
		}
		else {
#pragma omp parallel for if(ompCount > ompLimit)
			for (size_t j = 0; j < colsTemp; ++j){
				for ( size_t i = 0; i < rowsTemp; ++i){
					output_dataTemp[j * output_mrowsTemp + i] =  std::pow(dataTemp[j * mrowsTemp + i], pow1) * mult + add;
				}
			}
		}
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseOp(T add, T mult, T pow1){
		LaspMatrix<T> output;
		int err = eWiseOp(output, add, mult, pow1);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::eWiseDivM(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, T pow1, T pow2){
		if(otherMatrix.rows() != rows() || otherMatrix.cols() != cols()){
#ifndef NDEBUG
			cerr << "Error: Matrix dimensions must be equal for eWiseDivM" << endl;
#endif
			return INVALID_DIMENSIONS;
		}
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(cols(), rows(), false);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix, &output);
				device_eWiseDiv(params, dData(), otherMatrix.dData(), output.dData(), size(), pow1, pow2, rows(), mRows(), otherMatrix.mRows(), output.mRows());
				
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		output.transferToHost();
		
		output.resize(cols(), rows(), false);
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t mrowsTemp = mRows();
		T* dataTemp = data();
		
		size_t output_mrowsTemp = output.mRows();
		T* output_dataTemp = output.data();
		
		T* other_dataTemp = otherMatrix.data();
		size_t other_mrowsTemp = otherMatrix.mRows();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * colsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for( size_t j = 0; j < colsTemp; ++j){
			for( size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[j * output_mrowsTemp + i] = std::pow(dataTemp[j*mrowsTemp + i], pow1) / std::pow(other_dataTemp[j * other_mrowsTemp + i], pow2);
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseDivM(LaspMatrix<T>& otherMatrix, T pow1, T pow2){
		LaspMatrix<T> output;
		int err = eWiseDivM(otherMatrix, output, pow1, pow2);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	int LaspMatrix<T>::eWiseScale(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& outputMatrix, int d0, int d){
		if (device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &outputMatrix);
				return device_eWiseScale(params, dData(), outputMatrix.dData(), otherMatrix.dData(),rows(),cols(), d0);
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		
		for (int i=d0+1; i<=d; ++i){
			for (int j=0; j< rows(); j++){
				outputMatrix(j, i) = this->operator()((i-(d0+1)), j) * otherMatrix(j);
			}
		}
		
		return 0;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::eWiseScale(LaspMatrix<T>& otherMatrix, int d0, int d){
		LaspMatrix<T> output;
		int err = eWiseScale(otherMatrix, output, d0, d);
		
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		return output;
	}
	
	template<class T>
	T LaspMatrix<T>::trace(){
		LaspMatrix<T> d = diag();
		LaspMatrix<T> tr;
		d.colSum(tr);
		return tr(0);
	}
	
	template<class T>
	int LaspMatrix<T>::swap(){
		if(isSubMatrix()){
#ifndef NDEBUG
			cerr << "Cannot swap a sub-matrix" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		std::swap(_rows(), _cols());
		std::swap(_mRows(), _mCols());
		return MATRIX_SUCCESS;
	}
	
	
	template<class T>
	template<class ITER>
	int LaspMatrix<T>::gather(LaspMatrix<T>& output, ITER begin, ITER end){
		
		bool copy = output.key() == key();
		long size = end - begin;
		
		if(copy){
			cerr << "ERROR: Gather into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}
		
		if (device()){
			int* map = new int[size];
			std::copy(begin, end, map);
			LaspMatrix<int> mapMat(size, 1, map);
			return gather(output, mapMat);
		}
		else{
			output.transferToHost();
			output.resize(size, rows(), false);
			
			size_t ind = 0;
			
			size_t rowsTemp = rows();
			size_t colsTemp = cols();
			size_t mRowsTemp = mRows();
			size_t output_mrowsTemp = output.mRows();
			
			T* dataTemp = data();
			T* output_dataTemp = output.data();
			
			
			for (ITER i = begin; i != end; ++i, ++ind){
				for(size_t j = 0; j < rowsTemp; ++j){
					if(*i >= colsTemp){
						cerr << "ERROR: Map index for gather is out of bounds" << endl;
						return OUT_OF_BOUNDS;
					}
					
					output_dataTemp[ind * output_mrowsTemp + j] = dataTemp[(*i) * mRowsTemp + j];
				}
			}
			
			return MATRIX_SUCCESS;
		}
	}
	
	template<class T>
	int LaspMatrix<T>::gather(LaspMatrix<T>& output, LaspMatrix<int> map){
		int size = map.size();
		
		bool copy = output.key() == key();
		if(copy){
			cerr << "ERROR: Gather into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}
		if (device()){
			int error = MATRIX_SUCCESS;
			error += map.transferToDevice();
			error += output.transferToDevice();
			error += output.resize(size, rows(), false);
			
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_gather(params, map.dData(), dData(), output.dData(), rows(), mRows(), output.mRows(), size);
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		map.transferToHost();
		output.transferToHost();
		
		output.resize(size, rows(), false);
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		
		size_t rowsTemp = rows();
		size_t output_mrowsTemp = output.mRows();
		size_t mRowsTemp = mRows();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * size;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for (size_t ind = 0; ind < size; ++ind){
			size_t map_ind = map(ind);
			
			for(size_t j = 0; j < rowsTemp; ++j){
				
				output_dataTemp[ind * output_mrowsTemp + j] = dataTemp[map_ind * mRowsTemp + j];
			}
		}
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	int LaspMatrix<T>::gather(LaspMatrix<T>& output, vector<int>& map){
		if(device()){
			return gather(output, map.begin(), map.end());
		}
		
		size_t size = map.size();
		
		bool copy = output.key() == key();
		if(copy){
			cerr << "ERROR: Gather into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}
		
		transferToHost();
		output.transferToHost();
		
		output.resize(size, rows(), false);
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		
		size_t rowsTemp = rows();
		size_t output_mrowsTemp = output.mRows();
		size_t mRowsTemp = mRows();
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * size;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for (size_t ind = 0; ind < size; ++ind){
			size_t map_ind = map[ind];
			
			for(size_t j = 0; j < rowsTemp; ++j){
				
				output_dataTemp[ind * output_mrowsTemp + j] = dataTemp[map_ind * mRowsTemp + j];
			}
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	template<class ITER>
	LaspMatrix<T> LaspMatrix<T>::gather(ITER begin, ITER end){
		LaspMatrix<T> output;
		
		int err = gather(output, begin, end);
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		
		return output;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::gather(LaspMatrix<int> map){
		LaspMatrix<T> output;
		
		int err = gather(output, map);
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		
		return output;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::gather(vector<int>& map){
		LaspMatrix<T> output;
		
		int err = gather(output, map);
		if(err != MATRIX_SUCCESS){
			throw err;
		}
		
		return output;
	}
	
	
	
	template<class T>
	template<class ITER>
	int LaspMatrix<T>::gatherSum(LaspMatrix<T>& output, ITER begin, ITER end){
		if (output.rows() > 1 && output.cols() > 1) {
			cerr << "ERROR: Gather sum ouput must be a vector" << endl;
			return CANNOT_COMPLETE_OPERATION;
		}
		
		bool copy = output.key() == key();
		long size = end - begin;
		
		if(copy){
			cerr << "ERROR: Gather sum into same matrix not supported" << endl;
			return INVALID_LOCATION;
		}
		
		if (device()){
			int* map = new int[size];
			std::copy(begin, end, map);
			LaspMatrix<int> mapMat(size, 1, map);
			return gatherSum(output, mapMat);
		}
		else{
			if(output.size() < rows()){
				output.resize(1, rows(), false, true);
			}
			
			size_t ind = 0;
			
			size_t rowsTemp = rows();
			size_t colsTemp = cols();
			size_t mRowsTemp = mRows();
			
			T* dataTemp = data();
			T* output_dataTemp = output.data();
			size_t stride = output.rows() == 1 ? output.mRows() : 1;
			
			for(size_t j = 0; j < rowsTemp; ++j){
				for (ITER i = begin; i != end; ++i, ++ind){
					if(*i >= colsTemp){
						cerr << "ERROR: Map index for gather is out of bounds" << endl;
						return OUT_OF_BOUNDS;
					}
					
					output_dataTemp[j * stride] += dataTemp[(*i) * mRowsTemp + j];
				}
			}
			
			return MATRIX_SUCCESS;
		}
	}
	
	template<class T>
	int LaspMatrix<T>::gatherSum(LaspMatrix<T>& output, LaspMatrix<int> map){
		if (output.rows() > 1 && output.cols() > 1) {
			cerr << "ERROR: Gather sum ouput must be a vector" << endl;
			return CANNOT_COMPLETE_OPERATION;
		}
		
		size_t size = map.size();
		
		bool copy = output.key() == key();
		if(copy){
#ifndef NDEBUG
			cerr << "ERROR: Gather sum into same matrix not supported" << endl;
#endif
			return INVALID_LOCATION;
		}
		if (device()){
			int error = MATRIX_SUCCESS;
			error += map.transferToDevice();
			error += output.transferToDevice();
			if(output.size() < rows()){
				error += output.resize(size, 1, false, true);
			}
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &output);
				device_gatherSum(params, map.dData(), dData(), output.dData(), rows(), mRows(), output.mRows(), size, output.rows());
				return MATRIX_SUCCESS;
			}
		}
		
		transferToHost();
		map.transferToHost();
		output.transferToHost();
		
		if(output.size() < rows()){
			output.resize(1, rows(), false, true);
		}
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		
		size_t rowsTemp = rows();
		size_t output_mrowsTemp = output.mRows();
		size_t mRowsTemp = mRows();
		size_t stride = output.rows() == 1 ? output.mRows() : 1;
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * size;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < rowsTemp; ++j){
			for (size_t ind = 0; ind < size; ++ind){
				size_t map_ind = map(ind);
				output_dataTemp[j * stride] += dataTemp[map_ind * mRowsTemp + j];
			}
		}
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	int LaspMatrix<T>::gatherSum(LaspMatrix<T>& output, vector<int>& map){
		if(device()){
			return gatherSum(output, map.begin(), map.end());
		}
		
		if (output.rows() > 1 && output.cols() > 1) {
			cerr << "ERROR: Gather sum ouput must be a vector" << endl;
			return CANNOT_COMPLETE_OPERATION;
		}
		
		size_t size = map.size();
		
		bool copy = output.key() == key();
		if(copy){
#ifndef NDEBUG
			cerr << "ERROR: Gather sum into same matrix not supported" << endl;
#endif
			return INVALID_LOCATION;
		}
		
		transferToHost();
		output.transferToHost();
		
		if(output.size() < rows()){
			output.resize(1, rows(), false, true);
		}
		
		T* output_dataTemp = output.data();
		T* dataTemp = data();
		
		size_t rowsTemp = rows();
		size_t output_mrowsTemp = output.mRows();
		size_t mRowsTemp = mRows();
		size_t stride = output.rows() == 1 ? output.mRows() : 1;
		
#ifdef _OPENMP
		size_t ompCount = rowsTemp * size;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < rowsTemp; ++j){
			for (size_t ind = 0; ind < size; ++ind){
				size_t map_ind = map[ind];
				output_dataTemp[j * stride] += dataTemp[map_ind * mRowsTemp + j];
			}
		}
		
		return MATRIX_SUCCESS;
		
	}
	
	template<class T>
	int LaspMatrix<T>::contigify(LaspMatrix<int>& map){
		LaspMatrix<int> indMap(2, cols());
		
		int numCols = indMap.rows();
		int* colToPosPtr = &(indMap(0,0));
		int* posToColPtr = &(indMap(1,0));
		for (int i = 0; i < numCols ; ++i) {
			colToPosPtr[i] = i;
			posToColPtr[i] = i;
		}
		
		LaspMatrix<T> swapBuffer(1, rows());
		
		int error = 0;
		if (device()) {
			error += swapBuffer.transferToDevice();
		}
		
		if (error) {
			transferToHost();
			swapBuffer.transferToHost();
		}
		
		for (int i = 0; i < map.size(); ++i) {
			int targetPos = i;
			int targetCol = map(i);
			int colInTargetPos = posToColPtr[targetPos];
			int posOfTargetCol = colToPosPtr[targetCol];
			
			if (posOfTargetCol == targetPos) {
				continue;
			}
			
			swapBuffer.setCol(0, *this, targetPos);
			setCol(targetPos, *this, posOfTargetCol);
			setCol(posOfTargetCol, swapBuffer, 0);
			
			posToColPtr[targetPos] = targetCol;
			posToColPtr[posOfTargetCol] = colInTargetPos;
			colToPosPtr[targetCol] = targetPos;
			colToPosPtr[colInTargetPos] = posOfTargetCol;
		}
		
		map = indMap;
		
		return MATRIX_SUCCESS;
	}
	
	template <class T>
	int LaspMatrix<T>::revert(LaspMatrix<int>& map){
		LaspMatrix<int> indMap = map;
		
		int numCols = indMap.rows();
		int* colToPosPtr = &(indMap(0,0));
		int* posToColPtr = &(indMap(1,0));
		
		LaspMatrix<T> swapBuffer(1, rows());
		
		int error = 0;
		if (device()) {
			error += swapBuffer.transferToDevice();
		}
		
		if (error) {
			transferToHost();
			swapBuffer.transferToHost();
		}
		
		for (int i = 0; i < numCols; ++i) {
			int targetPos = i;
			int targetCol = i;
			int colInTargetPos = posToColPtr[targetPos];
			int posOfTargetCol = colToPosPtr[targetCol];
			
			if (posOfTargetCol == targetPos) {
				continue;
			}
			
			swapBuffer.setCol(0, *this, targetPos);
			setCol(targetPos, *this, posOfTargetCol);
			setCol(posOfTargetCol, swapBuffer, 0);
			
			posToColPtr[targetPos] = targetCol;
			posToColPtr[posOfTargetCol] = colInTargetPos;
			colToPosPtr[targetCol] = targetPos;
			colToPosPtr[colInTargetPos] = posOfTargetCol;
		}
		
		map = LaspMatrix<int>();
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::shuffle(LaspMatrix<int> &revert_map, LaspMatrix<int> &shuffle_map) {
		vector<int> indices(cols());
		for (int i = 0; i < indices.size(); ++i) {
			indices[i] = i;
		}
		
		random_shuffle(indices.begin(), indices.end());
		LaspMatrix<int> temp_map(indices);
		shuffle_map.operator=(temp_map.copy());
		int ret_val = contigify(temp_map);
		revert_map.operator=(temp_map);
		return ret_val;
	}
	
	template<class T>
	int LaspMatrix<T>::shuffle(LaspMatrix<int> &revert_map) {
		LaspMatrix<int> shuffle_map;
		return shuffle(revert_map, shuffle_map);
	}
	
	template<class T>
	int LaspMatrix<T>::shuffle(){
		LaspMatrix<int> map;
		return shuffle(map);
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::gatherMap(vector<int>& map){
		LaspMatrix<T> retVal(1, cols(), 0.0);
		
		int mapSize = map.size();
		T* dataTemp = retVal.data();
		
		for (int i = 0; i < mapSize; ++i) {
			dataTemp[map[i]] = 1.0;
		}
		
		return retVal;
	}
	
	template<class T>
	int LaspMatrix<T>::solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output, LaspMatrix<T>& LU, LaspMatrix<int>& ipiv){
		cerr << "Error: Solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::solve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output){
		cerr << "Error: Solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::solve(LaspMatrix<T>& otherMatrix){
		cerr << "Error: Solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	
	template<class T>
	int LaspMatrix<T>::chol(LaspMatrix<T>& output){
		cerr << "Error: Cholesky decomposition not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::chol(){
		cerr << "Error: Cholesky decomposition not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::cholSolve(LaspMatrix<T>& otherMatrix, LaspMatrix<T>& output){
		cerr << "Error: Cholesky solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::cholSolve(LaspMatrix<T>& otherMatrix){
		cerr << "Error: Cholesky solve not implemented for type!" << endl;
		throw METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	void LaspMatrix<T>::printMatrix(string name, int c, int r){
		bool transfer = device();
		transferToHost();
		
		cout.precision(16);
		if(c == 0)
			c = cols();
		if(r == 0)
			r = rows();
		
		if(!name.empty()){
			cout << name << ":" << endl;
		}
		for(int i=0; i < r; ++i){
			for (int j=0; j< c; ++j){
				cout << setw(20) << data()[j*(size_t)mRows()+i]  << " ";
			}
			cout << endl;
		}
		
		cout.precision(8);
		
		if(transfer){
#ifndef NDEBUG
			cerr << "Warning: printing device matrix requires expensive memory transfer";
#endif
			transferToDevice();
		}
	}
	
	template<class T>
	void LaspMatrix<T>::printInfo(string name){
		
		
		cout.precision(5);
		
		if(!name.empty()){
			cout << name << ":" << endl;
		}
		
		cout << "Size: (" << cols() << ", " << rows() << "), mSize: (" << cols() << ", " << rows() << ")" << endl;
		cout << "On device: " << device() << endl;
		
	}
	
	template<class T>
	int LaspMatrix<T>::addMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out){
		if (device()){
			int error = b.transferToDevice();
			error += out.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &out);
				device_addMatrix(params, dData(), b.dData(), out.dData(), rows(), cols(), mRows(), b.mRows(), out.mRows());
				return 0;
			}
		}
		
		transferToHost();
		b.transferToHost();
		out.transferToHost();
		
		size_t size = (size_t)rows()*(size_t)cols();
		for (size_t i = 0; i < size; ++i){
			out.data()[i] = data()[i] + b.data()[i];
		}
		
		return 0;
		
	}
	
	
	template<class T>
	int LaspMatrix<T>::subMatrix(LaspMatrix<T>& b, LaspMatrix<T>& out){
		if (device()){
			int error = b.transferToDevice();
			error += out.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &out);
				device_subMatrix(params, dData(), b.dData(), out.dData(), rows(), cols(), mRows(), b.mRows(), out.mRows());
				return 0;
			}
		}
		
		transferToHost();
		b.transferToHost();
		out.transferToHost();
		
		size_t size = (size_t)rows()*(size_t)cols();
		
		for (size_t i = 0; i < size; ++i){
			out.data()[i] = data()[i] - b.data()[i];
		}
		
		return 0;
	}
	
	
	
	template<class T>
	int LaspMatrix<T>::ger(LaspMatrix<T>& output, LaspMatrix<T> X, LaspMatrix<T> Y, T alpha){
		cerr << "Error: Ger not implemented for type!" << endl;
		return METHOD_NOT_IMPLEMENTED;
	}
	
	template<class T>
	int LaspMatrix<T>::ger(LaspMatrix<T> X, LaspMatrix<T> Y, T alpha){
		return ger(*this, X, Y, alpha);
	}
	
	
	//Hacky way of checking for the same types
	template<class T, class N>
	inline bool sameType(T x, N y){
		return false;
	}
	
	template<>
	inline bool sameType(float x, float y){
		return true;
	}
	
	template<>
	inline bool sameType(double x, double y){
		return true;
	}
	
	template<>
	inline bool sameType(int x, int y){
		return true;
	}
	
	
  	template<class T>
  	template<class N>
	LaspMatrix<N> LaspMatrix<T>::convert(bool mem){
		//Check that T and N are not already the same type
		T x = 1;
		N y = 1;
		if (sameType(x, y)){
			return *(reinterpret_cast<LaspMatrix<N>*>(this));
		}
		
		LaspMatrix<N> result;
		
		if(device()){
			int error = result.transferToDevice();
			error += result.resize(cols(), rows(), false);
			
			if(error == MATRIX_SUCCESS){
				//temporary hack, templating didn't work immediately moving on TODO
				DeviceParams params = context().setupOperation(this, this);
				device_convert(params, dData(), result.dData(), rows(), cols(), mRows());
				return result;
			}
		}
		
		transferToHost();
		
		if(mem){
			result.resize(mCols(), mRows());
		}
		
		result.resize(cols(), rows());
		
		N* output_dataTemp = result.data();
		T* dataTemp = data();
		
		size_t rowsTemp = rows();
		size_t colsTemp = cols();
		size_t output_mrowsTemp = result.mRows();
		size_t mRowsTemp = mRows();
		
		for(size_t i = 0; i < colsTemp; ++i){
			for(size_t j = 0; j < rowsTemp; ++j){
				output_dataTemp[i * output_mrowsTemp + j] = static_cast<N>(dataTemp[mRowsTemp * i + j]);
			}
		}
		
		return result;
	}
	
	template<class T>
	bool LaspMatrix<T>::operator==(LaspMatrix<T>& other){
		if(key() == other.key()){
			return true;
		}
		
		if(rows() != other.rows() || cols() != other.cols()){
			return false;
		}
		
		bool result = true;
		for(int i = 0; i < cols(); ++i){
			for(int j = 0; j < rows(); ++j){
				T epsilon = operator()(i, j) / 1000.0;
				if(abs(operator()(i, j) - other(i, j)) > epsilon){
					result = false;
				}
			}
		}
		
		return result;
	}
	
	template<class T>
	void LaspMatrix<T>::checksum(string name){
		T sum = 0;
		T xor_sum = 0;
		for(int i = 0; i < rows(); ++i){
			for(int j = 0; j < cols(); ++j){
				T elem = operator()(j, i);
				sum += elem;
			}
		}
		
		cout << name << ", sum: " << sum << ", xor: " << xor_sum << endl;
	}
	
	template<class T>
	template<class N>
	N* LaspMatrix<T>::getRawArrayCopy(){
		if(device()){
			transferToHost();
		}
		
		N* output = new N[rows() * cols()];
		
		for (size_t jj = 0; jj < cols(); jj++) {
			for (size_t ii = 0; ii < rows(); ii++) {
				output[(size_t)rows() * jj + ii] = static_cast<N>(operator()(jj, ii));
			}
		}
		
		return output;
	}
	
	template<>
	inline int LaspMatrix<float>::multiply(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& outputMatrix, bool transposeMe, bool transposeOther, float a, float b, int numRowsToSkip){

		int myRows = transposeMe ? cols() : rows();
		int myCols = transposeMe ? rows() : cols();
		int otherRows = transposeOther ? otherMatrix.cols() : otherMatrix.rows();
		int otherCols = transposeOther ? otherMatrix.rows() : otherMatrix.cols();
		
		if (myCols != (transposeOther ? otherRows : otherRows - numRowsToSkip)) {
			if (size() == 1) {
				return otherMatrix.multiply(this->operator()(0), outputMatrix);
			}
			
			if (otherMatrix.size() == 1) {
				return multiply(otherMatrix(0), outputMatrix);
			}
			
			cerr << "Error: Dimension mismatch in multiply" << endl;
			return INVALID_DIMENSIONS;
		}
		
		int numToSkip = 0;
		if (!transposeOther) {
			numToSkip = numRowsToSkip;
			numRowsToSkip = 0;
		}
		
		int m = myRows;
		int n = otherCols - numRowsToSkip;
		int k = myCols;
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		
		float alpha(a);
		float beta(b);
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			error += outputMatrix.resize(n, m, copy);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix);
				return device_sgemm(params, transposeMe, transposeOther, m, n, k, alpha, dData(), mRows(), otherMatrix.dData() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.dData(), outputMatrix.mRows());
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		outputMatrix.resize(n, m, copy, false, 0, true);
		host_sgemm(transposeMe, transposeOther, m, n, k, alpha, data(), mRows(), otherMatrix.data() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.data(), outputMatrix.mRows());
		
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::solve(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& output, LaspMatrix<float>& LU, LaspMatrix<int>& ipiv){
		if (cols() != rows() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in solve" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Apparently the cuda getrfbatched kernel is terrible for single solves
		transferToHost();
		
		output.resize(otherMatrix.cols(), otherMatrix.rows());
		
		int n = rows();
		int nrhs = otherMatrix.cols();
		
		int error = MATRIX_SUCCESS;
		
		LaspMatrix<float> A;
		
		if(device()){
			error += A.transferToDevice();
		}
		
		if (LU.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		
		int lda = A.mRows();
		LaspMatrix<int> ipivOut(1, n);
		LaspMatrix<float> B;
		
		if(device()){
			error += otherMatrix.transferToDevice();
			error += B.transferToDevice();
			error += ipivOut.transferToDevice();
		}
		
		if (output.key() == otherMatrix.key()){
			B = otherMatrix;
		} else {
			B.copy(otherMatrix);
		}
		
		int ldb = B.mRows();
		int info = 0;
		
		if(device() && error == MATRIX_SUCCESS){
			DeviceParams params = context().setupOperation(this, &otherMatrix);
			device_sgesv(params, n, nrhs, A.dData(), lda, ipivOut.dData(), B.dData(), ldb);
		} else {
			
			A.transferToHost();
			ipivOut.transferToHost();
			B.transferToHost();
			otherMatrix.transferToHost();
			
			info = host_sgesv(n, nrhs, A.data(), lda, ipivOut.data(), B.data(), ldb);
			
			if(info < 0){
#ifndef NDEBUG
				cerr << "Argument " << -info << " to gesv invalid!" << endl;
#endif
				return ARGUMENT_INVALID;
			} else if (info > 0){
#ifndef NDEBUG
				cerr << "Factor " << info << " in gesv is singular!" << endl;
#endif
				return CANNOT_COMPLETE_OPERATION;
			}
		}
		
		ipiv = ipivOut;
		output.operator=(B);
		LU = A;
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::solve(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& output){
		LaspMatrix<float> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, output, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<float>::solve(LaspMatrix<float>& otherMatrix){
		LaspMatrix<float> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, otherMatrix, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<float>::chol(){
		if (cols() != rows()) {
			cerr << "Error: Dimension mismatch in chol" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Cholesky decomposition not supported on device
		transferToHost();
		int info = 0;
		
		info = host_spotrf(false, rows(), data(), mRows());
		
		if(info < 0){
#ifndef NDEBUG
			cerr << "Argument " << -info << " to potrf invalid!" << endl;
#endif
			return ARGUMENT_INVALID;
		} else if (info > 0){
#ifndef NDEBUG
			cerr << "Leading minor " << info << " in potrf is not positive definite!" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::chol(LaspMatrix<float>& output){
		output.copy(*this);
		return output.chol();
	}
	
	template<>
	inline int LaspMatrix<float>::cholSolve(LaspMatrix<float>& otherMatrix){
		if (cols() != rows() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in cholSolve" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Cholesky solve not supported on device
		transferToHost();
		otherMatrix.transferToHost();
		int info = 0;
		
		info = host_spotrs(false, rows(), otherMatrix.cols(), data(), mRows(), otherMatrix.data(), otherMatrix.mRows());
		
		if(info < 0){
#ifndef NDEBUG
			cerr << "Argument " << -info << " to potrs invalid!" << endl;
#endif
			return ARGUMENT_INVALID;
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::cholSolve(LaspMatrix<float>& otherMatrix, LaspMatrix<float>& output){
		output.copy(otherMatrix);
		return cholSolve(output);
	}
	
	template<>
	inline int LaspMatrix<float>::ger(LaspMatrix<float>& output, LaspMatrix<float> X, LaspMatrix<float> Y, float alpha){
		int m = rows();
		int n = cols();
		
		int incx;
		if(!((X.rows() >= m && X.cols() == 1) || (X.cols() >= m && X.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (X.cols() == m){
			incx = X.mRows();
		} else {
			incx = 1;
		}
		
		int incy;
		if(!((Y.rows() >= n && Y.cols() == 1) || (Y.cols() >= n && Y.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (Y.cols() == n){
			incy = Y.mRows();
		} else {
			incy = 1;
		}
		
		LaspMatrix<float> A;
		if (output.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		if(A.device()){
			int error = MATRIX_SUCCESS;
			error += X.transferToDevice();
			error += Y.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(&A, &X, &Y);
				return device_sger(params, m, n, alpha, X.dData(), incx, Y.dData(), incy, A.dData(), A.mRows());
			}
		}
		
		transferToHost();
		X.transferToHost();
		Y.transferToHost();
		host_sger(m, n, alpha, X.data(), incx, Y.data(), incy, A.data(), A.mRows());
		
		output = A;
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline void LaspMatrix<double>::checksum(string name){
		double sum = 0;
		char xor_sum [sizeof(double) + 1];
		
		for (int k = 0; k < sizeof(double); ++k) {
			xor_sum[k] = 0;
		}
		
		for(int i = 0; i < rows(); ++i){
			for(int j = 0; j < cols(); ++j){
				double e = operator()(j, i);
				char xor_e [sizeof(double)];
				memcpy((void*)xor_e, (void*)&e,  sizeof(double));
				sum += operator()(j, i);
				for (int k = 0; k < sizeof(double); ++k) {
					xor_sum[k] = xor_sum[k] ^ xor_e[k];
				}
			}
		}
		
		cout << name << ", sum: " << setprecision(40) << sum << ", xor: ";
		for (int k = 0; k < sizeof(double); ++k) {
			cout << setbase(16) << (unsigned short) xor_sum[k] << " ";
		}
		
		cout << setbase(10) << setprecision(6);
		cout << endl;
	}
	
	template<>
	inline int LaspMatrix<double>::multiply(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& outputMatrix, bool transposeMe, bool transposeOther, double a, double b, int numRowsToSkip){
		
		int myRows = transposeMe ? cols() : rows();
		int myCols = transposeMe ? rows() : cols();
		int otherRows = transposeOther ? otherMatrix.cols() : otherMatrix.rows();
		int otherCols = transposeOther ? otherMatrix.rows() : otherMatrix.cols();
		
		if (myCols != (transposeOther ? otherRows : otherRows - numRowsToSkip)) {
			if (size() == 1) {
				return otherMatrix.multiply(this->operator()(0), outputMatrix);
			}
			
			if (otherMatrix.size() == 1) {
				return multiply(otherMatrix(0), outputMatrix);
			}
			
			cerr << "Error: Dimension mismatch in multiply" << endl;
			return INVALID_DIMENSIONS;
		}
		
		int numToSkip = 0;
		if (!transposeOther) {
			numToSkip = numRowsToSkip;
			numRowsToSkip = 0;
		}
		
		int m = myRows;
		int n = otherCols - numRowsToSkip;
		int k = myCols;
		
		//Resize output
		bool copy = (outputMatrix.key() == key()  || outputMatrix.key() == otherMatrix.key());
		
		//WHY?!?!?!?
		double alpha(a);
		double beta(b);
		
		if(device()){
			int error = MATRIX_SUCCESS;
			error += otherMatrix.transferToDevice();
			error += outputMatrix.transferToDevice();
			error += outputMatrix.resize(n, m, copy);
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(this, &otherMatrix);
				return device_dgemm(params, transposeMe, transposeOther, m, n, k, alpha, dData(), mRows(), otherMatrix.dData() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, outputMatrix.dData(), outputMatrix.mRows());
			}
		}
		
		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		
		LaspMatrix<double> result;
		if (copy) {
			result = LaspMatrix<double>(n, m);
		} else {
			result = outputMatrix;
			outputMatrix.resize(n, m);
		}
		
		host_dgemm(transposeMe, transposeOther, m, n, k, alpha, data(), mRows(), otherMatrix.data() + numRowsToSkip + numToSkip, otherMatrix.mRows(), beta, result.data(), result.mRows());
		
		outputMatrix.operator=(result);
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::solve(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& output, LaspMatrix<double>& LU, LaspMatrix<int>& ipiv){
		if (cols() != rows() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in solve" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Apparently the cuda getrfbatched kernel is terrible for single solves
		transferToHost();
		
		output.resize(otherMatrix.cols(), otherMatrix.rows());
		
		int n = rows();
		int nrhs = otherMatrix.cols();
		
		int error = MATRIX_SUCCESS;
		
		LaspMatrix<double> A;
		
		if(device()){
			error += A.transferToDevice();
		}
		
		if (LU.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		
		int lda = A.mRows();
		LaspMatrix<int> ipivOut(1, n);
		LaspMatrix<double> B;
		
		if(device()){
			error += otherMatrix.transferToDevice();
			error += B.transferToDevice();
			error += ipivOut.transferToDevice();
		}
		
		if (output.key() == otherMatrix.key()){
			B = otherMatrix;
		} else {
			B.copy(otherMatrix);
		}
		
		int ldb = B.mRows();
		int info = 0;
		
		if(device() && error == MATRIX_SUCCESS){
			
			DeviceParams params = context().setupOperation(this, &otherMatrix);
			device_dgesv(params, n, nrhs, A.dData(), lda, ipivOut.dData(), B.dData(), ldb);
			
		} else {
			
			A.transferToHost();
			ipivOut.transferToHost();
			B.transferToHost();
			otherMatrix.transferToHost();
			
			host_dgesv(n, nrhs, A.data(), lda, ipivOut.data(), B.data(), ldb);
			
			if(info < 0){
#ifndef NDEBUG
				cerr << "Argument " << -info << " to gesv invalid!" << endl;
#endif
				return ARGUMENT_INVALID;
			} else if (info > 0){
#ifndef NDEBUG
				cerr << "Factor " << info << " in gesv is singular!" << endl;
#endif
				return CANNOT_COMPLETE_OPERATION;
			}
		}
		
		ipiv = ipivOut;
		output.operator=(B);
		LU = A;
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::solve(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& output){
		LaspMatrix<double> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, output, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<double>::solve(LaspMatrix<double>& otherMatrix){
		LaspMatrix<double> LU;
		LaspMatrix<int> ipiv;
		
		return solve(otherMatrix, otherMatrix, LU, ipiv);
	}
	
	template<>
	inline int LaspMatrix<double>::chol(){
		if (cols() != rows()) {
			cerr << "Error: Dimension mismatch in chol" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Cholesky decomposition not supported on device
		transferToHost();
		int info = 0;
		
		info = host_dpotrf(false, rows(), data(), mRows());
		
		if(info < 0){
#ifndef NDEBUG
			cerr << "Argument " << -info << " to potrf invalid!" << endl;
#endif
			return ARGUMENT_INVALID;
		} else if (info > 0){
#ifndef NDEBUG
			cerr << "Leading minor " << info << " in potrf is not positive definite!" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::chol(LaspMatrix<double>& output){
		output.copy(*this);
		return output.chol();
	}
	
	template<>
	inline int LaspMatrix<double>::cholSolve(LaspMatrix<double>& otherMatrix){
		if (cols() != rows() || rows() != otherMatrix.rows()) {
			cerr << "Error: Dimension mismatch in cholSolve" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Cholesky solve not supported on device
		transferToHost();
		otherMatrix.transferToHost();
		int info = 0;
		
		info = host_dpotrs(false, rows(), otherMatrix.cols(), data(), mRows(), otherMatrix.data(), otherMatrix.mRows());
		
		if(info < 0){
#ifndef NDEBUG
			cerr << "Argument " << -info << " to potrs invalid!" << endl;
#endif
			return ARGUMENT_INVALID;
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::cholSolve(LaspMatrix<double>& otherMatrix, LaspMatrix<double>& output){
		output.copy(otherMatrix);
		return cholSolve(output);
	}
	
	template<>
	inline int LaspMatrix<double>::ger(LaspMatrix<double>& output, LaspMatrix<double> X, LaspMatrix<double> Y, double alpha){
		int m = rows();
		int n = cols();
		
		int incx;
		if(!((X.rows() >= m && X.cols() == 1) || (X.cols() >= m && X.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (X.cols() == m){
			incx = X.mRows();
		} else {
			incx = 1;
		}
		
		int incy;
		if(!((Y.rows() >= n && Y.cols() == 1) || (Y.cols() >= n && Y.rows() == 1))){
			cerr << "Error: Dimension mismatch in ger" << endl;
			return INVALID_DIMENSIONS;
		} else if (Y.cols() == n){
			incy = Y.mRows();
		} else {
			incy = 1;
		}
		
		LaspMatrix<double> A;
		if (output.key() == key()) {
			A = (*this);
		} else {
			A.copy(*this);
		}
		
		if(A.device()){
			int error = MATRIX_SUCCESS;
			error += X.transferToDevice();
			error += Y.transferToDevice();
			
			if(error == MATRIX_SUCCESS){
				DeviceParams params = context().setupOperation(&A, &X, &Y);
				return device_dger(params, m, n, alpha, X.dData(), incx, Y.dData(), incy, A.dData(), A.mRows());
			}
		}
		transferToHost();
		X.transferToHost();
		Y.transferToHost();
		host_dger(m, n, alpha, X.data(), incx, Y.data(), incy, A.data(), A.mRows());
		
		output = A;
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::add_outer(LaspMatrix<T> &otherMatrix, LaspMatrix<T> &outputMatrix) {
		if ((rows() > 1 && cols() > 1) || (otherMatrix.cols() > 1 && otherMatrix.rows() > 1)) {
			cerr << "Error: Dimension mismatch in add_outer" << endl;
			return INVALID_DIMENSIONS;
		}
		
		//Resize output
		bool copy = (outputMatrix.key() == key() || outputMatrix.key() == otherMatrix.key());
		
		LaspMatrix<T> outputFinal;
		if (!copy) {
			outputFinal = outputMatrix;
		}
		
		outputFinal.resize(otherMatrix.size(), size());
		
		transferToHost();
		otherMatrix.transferToHost();
		outputMatrix.transferToHost();
		outputFinal.transferToHost();
		
		size_t rowsTemp = outputFinal.rows();
		size_t colsTemp = outputFinal.cols();
		
		size_t stride = rows() == 1 ? mRows() : 1;
		size_t other_stride = otherMatrix.rows() == 1 ? otherMatrix.mRows() : 1;
		T* dataTemp = data();
		
		size_t output_mrowsTemp = outputFinal.mRows();
		T* output_dataTemp = outputFinal.data();
		
		T* other_dataTemp = otherMatrix.data();
		
#ifdef _OPENMP
		size_t ompCount = colsTemp * rowsTemp;
		size_t ompLimit = context().getOmpLimit();
#endif
		
#pragma omp parallel for if(ompCount > ompLimit)
		for(size_t j = 0; j < colsTemp; ++j){
			for (size_t i = 0; i < rowsTemp; ++i){
				output_dataTemp[output_mrowsTemp * j + i] = dataTemp[stride * i] + other_dataTemp[other_stride * j];
			}
		}
		
		outputMatrix = outputFinal;

		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::add_outer(LaspMatrix<T> &otherMatrix) {
		LaspMatrix<T> result;
		add_outer(otherMatrix, result);
		operator=(result);
		return 0;
	}
	
	template<class T>
	void distance2_nomul(LaspMatrix<T>& output, LaspMatrix<T> a, LaspMatrix<T>  b, LaspMatrix<T>  a_norm, LaspMatrix<T> b_norm){
		LaspMatrix<T> dist = output;
		
		a.multiply(b,dist,true,false, -2.0);
		
		int sizedAOnes = dist.cols();
		int sizedBOnes = dist.rows();
		
		LaspMatrix<T> onesA(1,sizedAOnes,1.0);
		LaspMatrix<T> onesB(1,sizedBOnes,1.0);
		
		dist.ger(onesB, b_norm);
		dist.ger(a_norm,onesA);
		
	}
	
	
	template<class T>
	void calc_rbf(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T> X2Param, LaspMatrix<T> Xnorm1Param, LaspMatrix<T> Xnorm2Param, T gamma){
		distance2_nomul(out, X1Param, X2Param, Xnorm1Param, Xnorm2Param);
		out.exp(gamma);
	}
	
	template<class T>
	void calc_lin(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T> X2Param){
		X1Param.multiply(X2Param,out,true,false);
		
	}
	
	template<class T>
	void calc_pol(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T>X2Param, T a, T c, T d){
		c = std::max(c, static_cast<T>(0)); //FIXME This stops NaN errors, but isn't quite correct
		LaspMatrix<T> out1(X1Param.cols(), X2Param.cols(), c);
		X1Param.multiply(X2Param, out1, true, false, a, 1.0);
		out1.eWiseOp(out,0,1,d);
	}
	
	template<class T>
	void calc_sigmoid(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T>X2Param, T a, T c){
		LaspMatrix<T> out1(X1Param.cols(), X2Param.cols(), c);
		X1Param.multiply(X2Param, out1, true, false, a, 1.0);
		out1.tanh();
		out = out1;
	}
	
	template<class T>
	void calc_ard(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T> X2Param, LaspMatrix<T> l, T scale, T gamma){
		LaspMatrix<T> M = LaspMatrix<T>::eye(X1Param.rows());
		
		//If l is a vector set it as the diagonal, otherwise assume it is square
		if(l.rows() == 1 || l.cols() == 1){
			for(int i = 0; i < l.size(); ++i){
				M(i, i) = 1.0 / l(i);
			}
		} else if (l.size() > 0) {
			M.eWiseDivM(l, M);
		} else {
			M.multiply(1.0 / gamma);
		}
		
		LaspMatrix<T> X1, X2, X1norm, X2norm;
		M.multiply(X1Param, X1);
		X1.colSqSum(X1norm);
		
		M.multiply(X2Param, X2);
		X2.colSqSum(X2norm);
		
		distance2_nomul(out, X1, X2, X1norm, X2norm);
		out.exp(0.5);
		out.multiply(scale*scale);
	}
	
	template<class T>
	void calc_exp(LaspMatrix<T>& out, LaspMatrix<T> X1Param, LaspMatrix<T>X2Param, T alpha, T beta){
		LaspMatrix<T> out1, out2;
		X1Param.add_outer(X2Param, out1);
		out1.add(beta);
		out1.pow(alpha);
		
		out = LaspMatrix<T>::ones(X2Param.cols(), X1Param.cols());
		out.eWiseDivM(out1, out);
	}
	
	template<class T>
	int LaspMatrix<T>::getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& Xnorm1, LaspMatrix<T>& X2, LaspMatrix<T>& Xnorm2, LaspMatrix<T>& l, bool mult, bool transMult, bool useGPU){
		//Check that the norms exist for RBF and SQDIST
		if((kernelOptions.kernel == RBF || kernelOptions.kernel == SQDIST) && Xnorm1.size() == 0){
			X1.colSqSum(Xnorm1);
		}
		
		if((kernelOptions.kernel == RBF || kernelOptions.kernel == SQDIST) && Xnorm2.size() == 0){
			X2.colSqSum(Xnorm2);
		}
		
		if(!unified() && context().getNumDevices() > 0 && kernelOptions.kernel != ARD && kernelOptions.kernel != SQDIST && kernelOptions.kernel != EXP && (useGPU || X1.device() || X2.device() || Xnorm1.device() || Xnorm2.device())){
			T *A, *Anorm, *B, *Bnorm, *out;
			size_t lda = X1.mRows(), aCols = X1.cols(), aRows = X1.rows();
			size_t ldb = X2.mRows(), bCols = X2.cols(), bRows = X2.rows();
			
			bool doKernel = !mult, trans = transMult, a_on_device = X1.device(), b_on_device = X2.device(), out_on_device = device();
			size_t aCPU = 1, bCPU = 1, aGPU = context().getNumDevices(), bGPU = 1, streams = 3, numDev = context().getNumDevices();
			
			if(!transMult){
				resize(bCols, aCols);
			} else {
				resize(bRows, aRows);
			}
			
			int ldOut = mRows();
			
			if(X1.device()){
				Xnorm1.transferToDevice();
				A = X1.dData();
				Anorm = Xnorm1.dData();
			} else {
				Xnorm1.transferToHost();
				A = X1.data();
				Anorm = Xnorm1.data();
			}
			
			if(X2.device()){
				Xnorm2.transferToDevice();
				B = X2.dData();
				Bnorm = Xnorm2.dData();
			} else {
				Xnorm2.transferToHost();
				B = X2.data();
				Bnorm = Xnorm2.data();
			}
			
			if(device()){
				out = dData();
			} else {
				out = data();
			}
			
			if(doKernel && kernelOptions.kernel == RBF && (Xnorm1.cols() != X1.cols() || Xnorm2.cols() != X2.cols() || Xnorm1.mRows() != 1 || Xnorm2.mRows() != 1)){
				cerr << "Improper inputs to getKernel!" << endl;
				return UNSPECIFIED_MATRIX_ERROR;
			}
			
			DeviceParams params = context().setupOperation(this);
			int error =  pinned_kernel_multiply(params, A, lda, aCols, Anorm, aRows, B, ldb, bCols, Bnorm, bRows, out, ldOut, kernelOptions, doKernel, aCPU, bCPU, aGPU, bGPU, streams, numDev, a_on_device, b_on_device, out_on_device, trans);
			if (error == 0) {
				return error;
			}
			
		}
		
		if(mult){
			bool trans1 = !transMult, trans2 = transMult;
			X1.multiply(X2, *this, trans1, trans2);
			
			return MATRIX_SUCCESS;
		}
		
		if (kernelOptions.kernel == RBF){
			calc_rbf(*this, X1, X2, Xnorm1, Xnorm2, static_cast<T>(kernelOptions.gamma));
		}
		if (kernelOptions.kernel == LINEAR){
			calc_lin(*this, X1, X2);
		}
		if (kernelOptions.kernel == POLYNOMIAL){
			calc_pol(*this, X1, X2, static_cast<T>(kernelOptions.gamma), static_cast<T>(kernelOptions.coef), static_cast<T>(kernelOptions.degree));
		}
		if (kernelOptions.kernel == SIGMOID){
			calc_sigmoid(*this, X1, X2, static_cast<T>(kernelOptions.gamma), static_cast<T>(kernelOptions.coef));
		}
		if (kernelOptions.kernel == ARD){
			calc_ard(*this, X1, X2, l, static_cast<T>(kernelOptions.scale), static_cast<T>(kernelOptions.gamma));
		}
		if (kernelOptions.kernel == SQDIST){
			distance2_nomul(*this, X1, X2, Xnorm1, Xnorm2);
		}
		if (kernelOptions.kernel == EXP){
			calc_exp(*this, X1, X2, static_cast<T>(kernelOptions.alpha), static_cast<T>(kernelOptions.beta));
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& Xnorm1, LaspMatrix<T>& X2, LaspMatrix<T>& Xnorm2, bool mult, bool transMult, bool useGPU){
		LaspMatrix<T> temp;
		return getKernel(kernelOptions, X1, Xnorm1, X2, Xnorm2, temp, mult, transMult, useGPU);
	}
	
	template<class T>
	int LaspMatrix<T>::getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& X2, LaspMatrix<T>& l, bool mult, bool transMult, bool useGPU){
		LaspMatrix<T> temp1, temp2;
		return getKernel(kernelOptions, X1, temp1, X2, temp2, l, mult, transMult, useGPU);
	}
	
	template<class T>
	int LaspMatrix<T>::getKernel(kernel_opt kernelOptions, LaspMatrix<T>& X1, LaspMatrix<T>& X2, bool mult, bool transMult, bool useGPU){
		LaspMatrix<T> temp1, temp2, temp3;
		return getKernel(kernelOptions, X1, temp1, X2, temp2, temp3, mult, transMult, useGPU);
	}
	
	template<class T>
	void LaspMatrix<T>::laspAlloc(size_t size){
		bool normalAlloc = true;
		if (unified()) {
#ifdef CUDA
#ifdef CUDA6
			if(cudaMallocManaged((void**)&_data(), size * sizeof(T)) == cudaSuccess){
				_dData() = _data()
				normalAlloc = false;
			} else {
				_unified() = false;
			}
#endif
#endif
			
		}
		
		if (normalAlloc){
			_data() = new T[size * sizeof(T)];
		}
	}
	
#ifdef CUDA
	template<class T>
	cudaError_t LaspMatrix<T>::laspCudaAlloc(void** ptr, size_t size){
		if (unified()) {
#ifdef CUDA6
			return cudaMallocManaged(ptr, size);
#endif
			
		} else {
			return cudaMalloc(ptr, size);
		}
	}
#endif
	
	template<class T>
	void LaspMatrix<T>::laspFree(T* ptr){
		if (unified()) {
#ifdef CUDA
#ifdef CUDA6
			cudaError_t err = cudaFree(ptr);
			if (err == cudaSuccess) {
				return;
			}
#endif
#endif
		}
		
		try {
			delete [] ptr;
		} catch (...) {}
	}
	
#ifndef CUDA
	
	template<class T>
	int LaspMatrix<T>::transfer(){
#ifndef NDEBUG
		cerr << "Warning: Transfer not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::transferToHost(){
#ifndef NDEBUG
		cerr << "Warning: Transfer to host not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::transferToDevice(){
#ifndef NDEBUG
		cerr << "Warning: Transfer to device not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::registerHost(){
#ifndef NDEBUG
		cerr << "Warning: Transfer to device not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetRow(size_t row, LaspMatrix<T>& other){
#ifndef NDEBUG
		cerr << "Warning: Device set row not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetRow(size_t row, LaspMatrix<T>& other, size_t otherRow){
#ifndef NDEBUG
		cerr << "Warning: Device set row not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetCol(size_t col, LaspMatrix<T>& other){
#ifndef NDEBUG
		cerr << "Warning: Device set col not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetCol(size_t col, LaspMatrix<T>& other, size_t otherCol){
#ifndef NDEBUG
		cerr << "Warning: Device set col not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceResize(size_t newCols, size_t newRows, bool copy, bool fill, T val){
#ifndef NDEBUG
		cerr << "Warning: Device resize not supported" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceCopy(LaspMatrix<T> &other){
#ifndef NDEBUG
		cerr << "Warning: Copy on device not supported, leaving data on host" << endl;
#endif
		return CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::deviceCopy(){
#ifndef NDEBUG
		cerr << "Warning: Copy on device not supported, leaving data on host" << endl;
#endif
		throw CANNOT_COMPLETE_OPERATION;
	}
	
	template<class T>
	void LaspMatrix<T>::freeData(){
		if (_data() != 0){
			delete [] _data();
		}
	}
	
#else
	
    template<class T>
	int LaspMatrix<T>::transfer(){
		//No explicit transfers if this matrix is in managed memory
		if(unified()){
			return MATRIX_SUCCESS;
		}
		
		if(isSubMatrix()){
#ifndef NDEBUG
			cerr << "Warning: Transferring a sub-matrix" << endl;
#endif
		}
		
		context().setupMemTransfer(this);
		if(device()) {
			if(_registered()){
				CUDA_CHECK(cudaHostUnregister(_data()));
				_registered() = false;
			} else {
				_data() = new T[(size_t)_mRows() * (size_t)_mCols()];
				if(dData() != 0){
					CUDA_CHECK(cudaMemcpy((void*)_data(), (void*)_dData(), (size_t)_mRows() * (size_t)_mCols() * sizeof(T), cudaMemcpyDeviceToHost));
					CUDA_CHECK(cudaFree((void*)_dData()));
				}
			}
			
			_device() = false;
			_dData() = 0;
			
		} else {
			if(mSize() != 0){
				if((static_cast<size_t>(mSize()) * sizeof(T)) >= context().getAvailableMemory()){
#ifndef NDEBUG
					cerr << "Not enough device memory! Requested: " << (static_cast<size_t>(mSize()) * sizeof(T)) << ", Available: " << context().getAvailableMemory() << endl;
#endif
					return UNSPECIFIED_MATRIX_ERROR;
				}
				CUDA_CHECK(cudaMalloc((void**)dData_, (size_t)_mRows() * (size_t)_mCols() * sizeof(T)));
				CUDA_CHECK(cudaMemcpy((void*)_dData(), (void*)_data(), (size_t)_mRows() * (size_t)_mCols() * sizeof(T), cudaMemcpyHostToDevice));
			} else {
				_dData() = 0;
			}
			
			delete [] _data();
			_data() = 0;
			_device() = true;
			
		}
		return MATRIX_SUCCESS;
	}
	
    template<class T>
	int LaspMatrix<T>::transferToDevice(){
		if(!device()){
			return transfer();
		} else{
			return MATRIX_SUCCESS;
		}
	}
	
    template<class T>
	int LaspMatrix<T>::transferToHost(){
		if(device()){
			return transfer();
		} else{
			return MATRIX_SUCCESS;
		}
	}
	
    template<class T>
	int LaspMatrix<T>::registerHost(){
		if(!device()){
			CUDA_CHECK(cudaHostRegister((void*)_data(), (size_t)_mRows() * (size_t)_mCols() * sizeof(T), cudaHostRegisterMapped));
			CUDA_CHECK(cudaHostGetDevicePointer((void**)&(_dData()), (void*)_data(), 0));
			_device() = true;
			_registered() = true;
		} else{
#ifndef NDEBUG
			cerr << "Warning: Data already on device or registered" << endl;
#endif
			return MATRIX_SUCCESS;
		}
	}
	
    template<class T>
	int LaspMatrix<T>::deviceResize(size_t newCols, size_t newRows, bool copy, bool fill, T val){
		if(!device()){
			return CANNOT_COMPLETE_OPERATION;
		}
		
		if ((newCols == 1 && rows() == 1 && cols() != 1) || (newRows == 1 && cols() == 1 && rows() != 1)){
			std::swap(_rows(), _cols());
			std::swap(_mRows(), _mCols());
		}
		
		if(mRows() < newRows || mCols() < newCols){
			context().setupMemTransfer(this);
			
			if ((max(newCols, cols()) * max(newRows, rows()) * 8) / 1000000000.0 > 1.0){
#ifndef NDEBUG
				cerr << "Allocating size: " << (max(newCols, cols()) * max(newRows, rows()) * sizeof(T)) / 1000000000.0 << " GB" << endl;
#endif
			}
			
			T* newptr = 0;
			CUDA_CHECK(laspCudaAlloc((void**)&newptr, (size_t)max(newCols, cols()) * (size_t)max(newRows, rows()) * sizeof(T)));
			
//			if(copy && rows() > 0 && cols() > 0 && newRows > 0 && newCols > 0 && newptr != 0 && dData() != 0){
//				CUDA_CHECK(cudaMemcpy((void*)newptr, (void*)dData(), mSize() * sizeof(T), cudaMemcpyDeviceToDevice));
//			}
//			
//			if(fill){
//				CUDA_CHECK(cudaMemset(newptr, 0, sizeof(T) * (size_t)newRows * (size_t)newCols));
//			}
			
			DeviceParams params = context().setupOperation(this);
			device_copy_fill(params, dData(), newptr, val, cols(), rows(), mRows(), newCols, newRows, newRows, fill);
			
			if(dData() != 0){
				if(_registered()){
					CUDA_CHECK(cudaHostUnregister(_data()));
					delete [] _data();
				} else {
					CUDA_CHECK(cudaFree((void*)dData()));
				}
			}
			
			_dData() = newptr;
			_mCols() = newCols;
			_mRows() = newRows;
			
			if (unified()) {
				_data() = _dData();
			}
		}
		
		if(fill){
			//eWiseOp(*this, val, 0, 0);
			DeviceParams params = context().setupOperation(this);
			device_copy_fill(params, dData(), dData(), val, cols(), rows(), mRows(), newCols, newRows, mRows(), fill);
		}
		
		_cols() = newCols;
		_rows() = newRows;
		
		return MATRIX_SUCCESS;
	}
	
	//Type agnostic wrapper for cublas<T>copy()
	template<class T>
	int setRowHelper(cublasHandle_t& handle, int n, const T* x, int incx, T* y, int incy){
#ifndef NDEBUG
		cerr << "Device to device row copy not supported for this type!" << endl;
#endif
		return	CUBLAS_STATUS_INVALID_VALUE;
	}
	
	template<>
	inline int setRowHelper(cublasHandle_t& handle, int n, const float* x, int incx, float* y, int incy){
		return cublasScopy(handle, n, x, incx, y, incy);
	}
	
	template<>
	inline int setRowHelper(cublasHandle_t& handle, int n, const double* x, int incx, double* y, int incy){
		return cublasDcopy(handle, n, x, incx, y, incy);
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetRow(size_t row, LaspMatrix<T>& other){
		if(!(other.cols() == 1 || (other.rows() == 1 && other.mRows() == 1))){
#ifndef NDEBUG
			cerr << "Warning: Device set row requires a contiguous memory vector" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		DeviceParams params = context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			T* rowPtr = dData() + row;
			T* vecPtr = other.data();
			CUBLAS_CHECK(cublasSetVector(cols(), sizeof(T), vecPtr, 1, rowPtr, mRows()));
		} else if(!device() && other.device()){
			T* rowPtr = data() + mRows() * row;
			T* vecPtr = other.dData();
			CUBLAS_CHECK(cublasGetVector(cols(), sizeof(T), vecPtr, 1, rowPtr, mRows()));
		} else if(device() && other.device()){
			T* rowPtr = dData() + mRows() * row;
			T* vecPtr = other.dData();
			CUBLAS_CHECK(setRowHelper(context().getCuBlasHandle(), cols(), vecPtr, 1, rowPtr, mRows()));
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetRow(size_t row, LaspMatrix<T>& other, size_t otherRow){
		
		DeviceParams params = context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			T* rowPtr = dData() + row;
			T* vecPtr = other.data() + otherRow;
			CUBLAS_CHECK(cublasSetVector(cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(!device() && other.device()){
			T* rowPtr = data() + mRows() * row;
			T* vecPtr = other.dData() + otherRow;
			CUBLAS_CHECK(cublasGetVector(cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(device() && other.device()){
			T* rowPtr = dData() + mRows() * row;
			T* vecPtr = other.dData() + otherRow;
			CUBLAS_CHECK(setRowHelper(context().getCuBlasHandle(), cols(), vecPtr, other.mRows(), rowPtr, mRows()));
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceSetCol(size_t col, LaspMatrix<T>& other){
		if(!(other.cols() == 1 || (other.rows() == 1 && other.mRows() == 1))){
#ifndef NDEBUG
			cerr << "Warning: Device set col requires a contiguous memory vector" << endl;
#endif
			return CANNOT_COMPLETE_OPERATION;
		}
		
		context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			T* colPtr = dData() + mRows() * col;
			T* vecPtr = other.data();
			CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyHostToDevice));
		} else if(!device() && other.device()){
			T* colPtr = data() + mRows() * col;
			T* vecPtr = other.dData();
			CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToHost));
		} else if(device() && other.device()){
			T* colPtr = dData() + mRows() * col;
			T* vecPtr = other.dData();
			CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToDevice));
		}
		
		return MATRIX_SUCCESS;
	}
	
	
	template<class T>
	int LaspMatrix<T>::deviceSetCol(size_t col, LaspMatrix<T>& other, size_t otherCol){
		
		context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			T* colPtr = dData() + mRows() * col;
			T* vecPtr = other.data() + other.mRows() * otherCol;
			CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyHostToDevice));
		} else if(!device() && other.device()){
			T* colPtr = data() + mRows() * col;
			T* vecPtr = other.dData() + other.mRows() * otherCol;
			CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToHost));
		} else if(device() && other.device()){
			T* colPtr = dData() + mRows() * col;
			T* vecPtr = other.dData() + other.mRows() * otherCol;
			CUDA_CHECK(cudaMemcpy(colPtr, vecPtr, rows() * sizeof(T), cudaMemcpyDeviceToDevice));
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	int LaspMatrix<T>::deviceCopy(LaspMatrix<T> &other){
		if(!device() && !other.device()){
			return CANNOT_COMPLETE_OPERATION;
		}
		
		context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			T* rowPtr = dData();
			T* vecPtr = other.data();
			CUBLAS_CHECK(cublasSetMatrix(rows(), cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(!device() && other.device()){
			T* rowPtr = data();
			T* vecPtr = other.dData();
			CUBLAS_CHECK(cublasGetMatrix(rows(), cols(), sizeof(T), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(device() && other.device()){
			T* rowPtr = dData();
			T* vecPtr = other.dData();
			DeviceParams params = context().setupOperation(&other, this);
			return device_ewiseOp(params, vecPtr, rowPtr, other.size(), 0, 1, 1, other.rows(), other.mRows(), mRows());
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<float>::deviceCopy(LaspMatrix<float> &other){
		if(!device() && !other.device()){
			return CANNOT_COMPLETE_OPERATION;
		}
		
		context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			float* rowPtr = dData();
			float* vecPtr = other.data();
			CUBLAS_CHECK(cublasSetMatrix(rows(), cols(), sizeof(float), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(!device() && other.device()){
			float* rowPtr = data();
			float* vecPtr = other.dData();
			CUBLAS_CHECK(cublasGetMatrix(rows(), cols(), sizeof(float), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(device() && other.device()){
			float* rowPtr = dData();
			float* vecPtr = other.dData();
			float alpha = 1, beta = 0;
			DeviceParams params = context().setupOperation(&other, this);
			CUBLAS_CHECK(cublasSgeam(params.context.getCuBlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, rows(), cols(), &alpha, other.dData(), other.mRows(), &beta, other.dData(), other.mRows(), dData(), mRows()));
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<>
	inline int LaspMatrix<double>::deviceCopy(LaspMatrix<double> &other){
		if(!device() && !other.device()){
			return CANNOT_COMPLETE_OPERATION;
		}
		
		context().setupMemTransfer(&other, this);
		
		if(device() && !other.device()){
			double* rowPtr = dData();
			double* vecPtr = other.data();
			CUBLAS_CHECK(cublasSetMatrix(rows(), cols(), sizeof(double), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(!device() && other.device()){
			double* rowPtr = data();
			double* vecPtr = other.dData();
			CUBLAS_CHECK(cublasGetMatrix(rows(), cols(), sizeof(double), vecPtr, other.mRows(), rowPtr, mRows()));
		} else if(device() && other.device()){
			double* rowPtr = dData();
			double* vecPtr = other.dData();
			double alpha = 1, beta = 0;
			DeviceParams params = context().setupOperation(&other, this);
			CUBLAS_CHECK(cublasDgeam(params.context.getCuBlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, rows(), cols(), &alpha, other.dData(), other.mRows(), &beta, other.dData(), other.mRows(), dData(), mRows()));
		}
		
		return MATRIX_SUCCESS;
	}
	
	template<class T>
	LaspMatrix<T> LaspMatrix<T>::deviceCopy(){
		if(!device()){
			throw CANNOT_COMPLETE_OPERATION;
		}
		
		LaspMatrix<T> result;
		
		result._rc() = 1;
		result._subrc() = 1;
		result._rowOffset() = 0;
		result._colOffset() = 0;
		result._rowEnd() = 0;
		result._colEnd() = 0;
		result._rows() = rows();
		result._cols() = cols();
		result._mRows() = mRows();
		result._mCols() = mCols();
		result._data() = 0;
		result._device() = true;
		result._key() = context().getNextKey();
		
		context().setupMemTransfer(this, &result);
		CUDA_CHECK_THROW(laspCudaAlloc((void**)result.dData_, mSize() * sizeof(T)));
		CUDA_CHECK_THROW(cudaMemcpy((void*)result._dData(), (void*)_dData(), mSize() * sizeof(T), cudaMemcpyDeviceToDevice));
		
		if (unified()) {
			_data() = _dData();
		}
		
		return result;
	}
	
	template<class T>
	void LaspMatrix<T>::freeData(){
		if (!device() && _data() != 0){
			delete [] _data();
		} else if (dData() != 0){
			if(_registered()){
				CUDA_CHECK_THROW(cudaHostUnregister(_data()));
				delete [] _data();
			} else {
				CUDA_CHECK_THROW(cudaFree((void*)_dData()));
			}
		}
	}
	
#endif
	
}

#endif
