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

#include "lasp_matrix.h"

extern "C"{
	void dgemm_(char* TRANSA, char* TRANSB, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
	void sgemm_(char* TRANSA, char* TRANSB, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
	void dger_(int* m, int* n, double* alpha, double* x, int* incx, double* y, int* incy, double* a, int* lda);
	void sger_(int* m, int* n, float* alpha, float* x, int* incx, float* y, int* incy, float* a, int* lda);
	void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
	void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);
	void dpotrf_(char* UPLO, int* n, double* a, int* lda, int* info);
	void spotrf_(char* UPLO, int* n, float* a, int* lda, int* info);
	void dpotrs_(char* UPLO, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, int* info);
	void spotrs_(char* UPLO, int* n, int* nrhs, float* a, int* lda, float* b, int* ldb, int* info);
	void dgecon_(char* NORM, int* n, double* a, int* lda, double* aNorm, double* rcond, double* work, int* iwork, int* info);
	void sgecon_(char* NORM, int* n, float* a, int* lda, float* aNorm, float* rcond, float* work, int* iwork, int* info);
	double dlange_(char* NORM, int* m, int*n, double* a, int* lda, double* work);
	float slange_(char* NORM, int* m, int*n, float* a, int* lda, float* work);
}
				 
namespace lasp{
	
#ifndef CUDA
	DeviceContext* DeviceContext::instance_ = 0;
	
	//From John d. Cook ( http://www.johndcook.com/cpp_phi.html )
	double cdf(double x) {
		// constants
		double a1 =  0.254829592;
		double a2 = -0.284496736;
		double a3 =  1.421413741;
		double a4 = -1.453152027;
		double a5 =  1.061405429;
		double p  =  0.3275911;
		
		// Save the sign of x
		int sign = 1;
		if (x < 0)
			sign = -1;
		x = fabs(x)/sqrt(2.0);
		
		// A&S formula 7.1.26
		double t = 1.0/(1.0 + p*x);
		double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
		
		return 0.5*(1.0 + sign*y);
	}
	
	float cdf(float x) {
		return (float) cdf((double) x);
	}
	
	double pdf(double x) {
		const double pi = 3.1415926535897;
		return (1.0 / sqrt(2 * pi)) * exp(-((x*x)/2));
	}
	
	float pdf(float x) {
		return (float) pdf((double) x);
	}
	
#endif
	
	int host_dgemm(bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc){
		char Atran = transa ? 't' : 'n';
		char Btran = transb ? 't' : 'n';
		
		dgemm_(&Atran, &Btran, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		
		return BLAS_SUCCESS;
	}
	
	int host_sgemm(bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float beta, float* c, int ldc){
		char Atran = transa ? 't' : 'n';
		char Btran = transb ? 't' : 'n';
		
		sgemm_(&Atran, &Btran, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		
		return BLAS_SUCCESS;
	}
	
	
	int host_dger(int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda){
		dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
		
		return BLAS_SUCCESS;
	}
	
	int host_sger(int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda){
		
		sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
		
		return BLAS_SUCCESS;
	}
	
	int host_dgesv(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, double* cond){
		int info = 0;
		double norm = 0;
		
		if (cond != 0) {
			norm = dlange_("1", &n, &n, a, &lda, cond);
		}
		
		dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
		
		if (info != 0) {
			return info;
		}
		
		//Check condition of matrix, useful for hessian
		if (cond != 0) {
			double* work = new double[4*n];
			int* iwork = new int[n];
		
			dgecon_("1", &n, a, &lda, &norm, cond, work, iwork, &info);
			
			delete [] work;
			delete [] iwork;
		}
		
		return MATRIX_SUCCESS;
	}
	
	int host_sgesv(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb, float* cond){
		int info = 0;
		float norm = 0;
		
		if (cond != 0) {
			norm = slange_("1", &n, &n, a, &lda, cond);
		}
		
		sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
		
		if (info != 0) {
			return info;
		}
		
		//Check condition of matrix, useful for hessian
		if (cond != 0) {
			float* work = new float[4*n];
			int* iwork = new int[n];
			
			sgecon_("1", &n, a, &lda, &norm, cond, work, iwork, &info);
			
			delete [] work;
			delete [] iwork;
		}
		
		return MATRIX_SUCCESS;
	}
	
	int host_dpotrf(bool upper, int n, double* a, int lda){
		char up = upper ? 'U' : 'L';
		int info = 0;
		
		dpotrf_(&up, &n, a, &lda, &info);
		
		return info;
	}
	
	int host_spotrf(bool upper, int n, float* a, int lda){
		char up = upper ? 'U' : 'L';
		int info = 0;
		
		spotrf_(&up, &n, a, &lda, &info);
		
		return info;
	}
	
	int host_dpotrs(bool upper, int n, int nrhs, double* a, int lda, double* b, int ldb){
		char up = upper ? 'U' : 'L';
		int info = 0;
		
		dpotrs_(&up, &n, &nrhs, a, &lda, b, &ldb, &info);
		
		return info;
	}
	
	int host_spotrs(bool upper, int n, int nrhs, float* a, int lda, float* b, int ldb){
		char up = upper ? 'U' : 'L';
		int info = 0;
		
		spotrs_(&up, &n, &nrhs, a, &lda, b, &ldb, &info);
		
		return info;
	}
	
	
#ifndef CUDA
	int device_dgesv(DeviceParams params, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb){
		cerr << "Device blas calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_sgesv(DeviceParams params, int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb){
		cerr << "Device blas calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_dgemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, double alpha, double* a, int lda, double * b, int ldb, double beta, double* c, int ldc){
		cerr << "Device blas calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_sgemm(DeviceParams params, bool transa, bool transb, int m, int n, int k, float alpha, float* a, int lda, float * b, int ldb, float beta, float* c, int ldc){
		cerr << "Device blas calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	
	int device_dger(DeviceParams params, int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda){
		cerr << "Device blas calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_sger(DeviceParams params, int m, int n, float alpha, float* x, int incx, float* y, int incy, float* a, int lda){
		cerr << "Device blas calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_colSqSum(DeviceParams params, float* A, size_t n, size_t features, float* result, float scalar, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_colSqSum(DeviceParams params, double* A, size_t n, size_t features, double* result, double scalar, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_colSum(DeviceParams params, float* A, size_t n, size_t features, float* result, float scalar, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_colSum(DeviceParams params, double* A, size_t n, size_t features, double* result, double scalar, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_ewiseOp(DeviceParams params, float* in, float* out, size_t length, float mult, float add, float pow1, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_ewiseOp(DeviceParams params, double* in, double* out, size_t length, double mult, double add, double pow1, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_ewiseOp(DeviceParams params, int* in, int* out, size_t length, int mult, int add, int pow1, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_copy_fill(DeviceParams params, float* in, float* out, float val, size_t cols, size_t rows, size_t mRows, size_t out_cols, size_t out_rows, size_t out_mRows, bool fill){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_copy_fill(DeviceParams params, double* in, double* out, double val, size_t cols, size_t rows, size_t mRows, size_t out_cols, size_t out_rows, size_t out_mRows, bool fill){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_copy_fill(DeviceParams params, int* in, int* out, int val, size_t cols, size_t rows, size_t mRows, size_t out_cols, size_t out_rows, size_t out_mRows, bool fill){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_colWiseMult(DeviceParams params, float* mat, float* out, float* vec, size_t rows, size_t cols, size_t mRows, size_t out_mRows, size_t vec_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_colWiseMult(DeviceParams params, double* mat, double* out, double* vec, size_t rows, size_t cols, size_t mRows, size_t out_mRows, size_t vec_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_rowWiseMult(DeviceParams params, float* mat, float* out, float* vec, size_t rows, size_t cols, size_t mRows, size_t out_mRows, size_t vec_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_rowWiseMult(DeviceParams params, double* mat, double* out, double* vec, size_t rows, size_t cols, size_t mRows, size_t out_mRows, size_t vec_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_eWiseDiv(DeviceParams params, float* in1, float* in2, float* out, size_t length, float pow1, float pow2, size_t rows, size_t in1_mRows, size_t in2_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_eWiseDiv(DeviceParams params, double* in1, double* in2, double* out, size_t length, double pow1, double pow2, size_t rows, size_t in1_mRows, size_t in2_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_eWiseMult(DeviceParams params, float* in1, float* in2, float* out, size_t length, float pow1, float pow2, size_t rows, size_t in1_mRows, size_t in2_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_eWiseMult(DeviceParams params, double* in1, double* in2, double* out, size_t length, double pow1, double pow2, size_t rows, size_t in1_mRows, size_t in2_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_gather(DeviceParams params, int* map, float* src, float* dst, size_t rows, size_t mRows, size_t out_mRows, size_t mapSize){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_gather(DeviceParams params, int* map, double* src, double* dst, size_t rows, size_t mRows, size_t out_mRows, size_t mapSize){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_exp(DeviceParams params, float* in, float* out, size_t n, size_t rows, size_t mRows, size_t out_mRows, float gamma){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	
	int device_exp(DeviceParams params, double* in, double* out, size_t n, size_t rows, size_t mRows, size_t out_mRows, double gamma){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_tanh(DeviceParams params, float* in, float* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_tanh(DeviceParams params, double* in, double* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_log(DeviceParams params, float* in, float* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_log(DeviceParams params, double* in, double* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_normCDF(DeviceParams params, double* in, double* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_normCDF(DeviceParams params, float* in, float* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_normPDF(DeviceParams params, double* in, double* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_normPDF(DeviceParams params, float* in, float* out, size_t n, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_chooseNextHelper(DeviceParams params, float* d_x, int d_xInd, float* g, float* h, int select, float* d_out_minus1, float* dK2, int dK2rows, int dK2cols){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	int device_chooseNextHelper(DeviceParams params, double* d_x, int d_xInd, double* g, double* h, int select, double* d_out_minus1, double* dK2, int dK2rows, int dK2cols){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_addMatrix(DeviceParams params, double* a, double* b, double* out, size_t rows, size_t cols, size_t a_mRows, size_t b_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_subMatrix(DeviceParams params, double* a, double* b, double* out, size_t rows, size_t cols, size_t a_mRows, size_t b_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_addMatrix(DeviceParams params, float* a, float* b, float* out, size_t rows, size_t cols, size_t a_mRows, size_t b_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_subMatrix(DeviceParams params, float* a, float* b, float* out, size_t rows, size_t cols, size_t a_mRows, size_t b_mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_transpose(DeviceParams params, float* in, float* out, size_t cols, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_transpose(DeviceParams params, double* in, double* out, size_t cols, size_t rows, size_t mRows, size_t out_mRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_gatherSum(DeviceParams params, int* map, float* src, float* dst, size_t rows, size_t mRows, size_t out_mRows, size_t mapSize, size_t outputRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	int device_gatherSum(DeviceParams params, int* map, double* src, double* dst, size_t rows, size_t mRows, size_t out_mRows, size_t mapSize, size_t outputRows){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	template<class T>
	int pinned_kernel_multiply(DeviceParams params, T* A, size_t lda, size_t aCols, T* aNorm, size_t aRows, T* B, size_t ldb, size_t bCols, T* bNorm, size_t bRows, T* Out, size_t ldOut, kernel_opt kernelOptions, bool doKernel, size_t a_cpuBlocks, size_t b_cpuBlocks, size_t a_gpuBlocks, size_t b_gpuBlocks, size_t num_streams_input, size_t num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose){
		cerr << "Device kernel calls not supported!" << endl;
		return CONTEXT_ERROR;
	}
	
	template
	int pinned_kernel_multiply<float>(DeviceParams params, float* A, size_t lda, size_t aCols, float* aNorm, size_t aRows, float* B, size_t ldb, size_t bCols, float* bNorm, size_t bRows, float* Out, size_t ldOut, kernel_opt kernelOptions, bool doKernel, size_t a_cpuBlocks, size_t b_cpuBlocks, size_t a_gpuBlocks, size_t b_gpuBlocks, size_t num_streams_input, size_t num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose);
	
	template
	int pinned_kernel_multiply<double>(DeviceParams params, double* A, size_t lda, size_t aCols, double* aNorm, size_t aRows, double* B, size_t ldb, size_t bCols, double* bNorm, size_t bRows, double* Out, size_t ldOut, kernel_opt kernelOptions, bool doKernel, size_t a_cpuBlocks, size_t b_cpuBlocks, size_t a_gpuBlocks, size_t b_gpuBlocks, size_t num_streams_input, size_t num_device_input, bool a_on_device, bool b_on_device, bool out_on_device, bool transpose);
	
#endif
	
}

