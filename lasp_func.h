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


#ifndef _LaspFunc_h
#define _LaspFunc_h

#include "lasp_matrix.h"
#include <functional>
#include <tuple>
#include <sstream>
#include <memory>

/*

 This file defines simple functions and operators for matrices. It uses a lazy evaluation scheme for evaluating expressions.
 
 The functions/operators defined so far are:
	"+"			- Add matrix or constant.
	"-"			- Subtract matrix or constant (also unary negate)
	"*"			- Matrix multiply or multiply by constant
	"/"			- Solve system of linear equations or divide by constant
	"+="		- Add and assign
	"-="		- Subtract and assign
	"*="		- Multiply and assign
	"/="		- Divide and assign
	"++"		- Element-wise increment
	"--"		- Element-wise decrement
 	"t"			- Transpose
	"mul"		- Element-wise multiply
	"div"		- Element-wise divide
	"max"		- Max element
	"min"		- Min element
	"trace"		- Trace
	"diag"		- Matrix diagonal (as column vector)
	"sel"		- Subselect part of matrix
	"csel"		- Subselect columns of a matrix
	"rsel"		- Subselect rows of a matrix
	"log"		- Element-wise log
	"exp"		- Element-wise exp
	"tanh"		- Element-wise tanh
	"pow"		- Element-wise exponent
	"sqrt"		- Element-wise square root
	"log"		- Element-wise log
	"pdf"		- Element-wise normal pdf
	"cdf"		- Element-wise normal cdf
	"sum"		- Sum of elements
	"csum"		- Column-wise sum
	"rsum"		- Row-wise sum
	"norm"		- Column-wise 2-norm
	"inv"		- Matrix inverse
	"copy"		- Copy a matrix
	"device"	- Move to device
	"host"		- Move to host
	"print"		- Print a matrix
	"info"		- Print matrix info
	"eye"		- Generate an identity matrix
	"ones"		- Generate a matrix of ones
	"zeros"		- Generate a matrix of zeros
	"rand"		- Generate a random matrix
	"hcat"		- Concatenate arbitrarily many matrices horizontally
	"vcat"		- Concatenate arbitrarily many matrices vertically
	"der"		- Get derivate of expression with respect to a matrix
 
 Possible other functions (not implemented):
	"chol"		- Cholesky decomposition
	"apply"		- Apply a function to each element of a matrix
	"capply"	- Apply a function to each column of a matrix
	"rapply"	- Apply a function to each row of a matrix
	"binop"		- Combine two matrices with a binary operator
	"reduce"	- Reduce matrix to a single value with a binary operator
	"creduce"	- Column-wise reduce
	"rreduce"	- Row-wise reduce
	"cgather"	- Gather columns of a matrix
	"rgather"	- Gather rows of a matrix
	"convert"	- Convert matrix to different type
	"kernel"	- Call getKernel
	"del"		- Clear matrix memory
 */

namespace lasp {
	
	enum oper_type { SIMPLE, MULT, TRANSPOSE, SOLVE, VEC, VAL };
	
	//Generates a matrix from a string in matlab style
	template<class T>
	LaspMatrix<T> make_matrix(string input){
		stringstream stream(input);
		
		int longest = 0;
		vector<vector<T> > mat;
		mat.push_back(vector<T>());
		string temp;
		
		while (true) {
			T val = 0;
			if (stream >> val) {
				mat.back().push_back(val);
			}
			else{
				stream.clear();
				
				longest = std::max((int)longest, (int)mat.back().size());
				if (stream >> temp) {
					mat.push_back(vector<T>());
				}
				else {
					break;
				}
			}
		}
		
		LaspMatrix<T> output(mat.size(), longest, (T)0);
		
		for(int i = 0; i < mat.size(); ++i) {
			for (int j = 0; j < mat[i].size(); ++j) {
				output(i, j) = mat[i][j];
			}
		}
		
		return output;
	}
	
	//Base class for lazy expression evaluation
	template <class T>
	struct EvalBase {
		
		bool has_matrix;
		bool is_matrix;
		bool has_parent;
		bool dims_cached;
		bool sub_matrix;
		
		int start_col;
		int start_row;
		int end_col;
		int end_row;
		
		int sub_col_start;
		int sub_row_start;
		int sub_col_end;
		int sub_row_end;
		
		size_t cached_size;
		pair<size_t, size_t> cached_dims;
		
		LaspMatrix<T> matrix;
		LaspMatrix<T> full_matrix;
		
		EvalBase<T>* parent;
		vector<shared_ptr<EvalBase<T> > > children;
		vector<LaspMatrix<T> > child_mats;
		vector<T> constants;
		
		EvalBase() : has_matrix(false), is_matrix(false), has_parent(false), dims_cached(false), sub_matrix(false), start_col(-1), start_row(-1), sub_col_end(-1), sub_row_end(-1), cached_size(0) {}
		
		EvalBase(LaspMatrix<T> input) : has_matrix(true), is_matrix(true), has_parent(false), dims_cached(false), sub_matrix(false), start_col(-1), start_row(-1), sub_col_end(-1), sub_row_end(-1), cached_size(0) {
			this->matrix = input;
			this->full_matrix = input;
		}
		
		//Defines the type of operation for preprocessing
		virtual oper_type type() {
			return SIMPLE;
		}
		
		//Propagates down size requirements for output
		// Note: -1 is a special case for ends, signifying end of matrix
		void set_submatrix(int start_c, int start_r, int end_c, int end_r, bool has_req){
			this->dims_cached = false;
			
			if (this->sub_matrix && !has_req) {
				this->start_col = this->sub_col_start;
				this->start_row = this->sub_row_start;
				this->end_col = this->sub_col_end;
				this->end_row = this->sub_row_end;
			}
			else if (has_req && !this->sub_matrix) {
				this->start_col = start_c;
				this->start_row = start_r;
				this->end_col = end_c;
				this->end_row = end_r;
			}
			else if (has_req && this->sub_matrix) {
				this->start_col = start_c + this->sub_col_start;
				this->start_row = start_r + this->sub_row_start;
				this->end_col = end_c == -1 ? this->sub_col_end : this->sub_col_start + end_c;
				this->end_row = end_r == -1 ? this->sub_row_end : this->sub_row_start + end_r;
			}
			else {
				this->start_col = -1;
				this->start_row = -1;
				this->end_col = -1;
				this->end_row = -1;
			}
			
			has_req = has_req || this->sub_matrix;
			
			if (this->has_matrix && has_req){
				int col = this->end_col == -1 ? this->full_matrix.cols() : this->end_col;
				int row = this->end_row == -1 ? this->full_matrix.rows() : this->end_row;
				this->matrix = this->full_matrix(this->start_col, this->start_row, col, row);
				return;
			}
			
			this->set_children_submatrices(this->start_col, this->start_row, this->end_col, this->end_row, has_req);
		}
		
		void set_submatrix() {
			this->set_submatrix(-1, -1, -1, -1, false);
		}
		
		//Operators for subselecting part of an expression result
		void operator()(int index){
			pair<size_t, size_t> dims = this->get_dims();
			operator()(index / dims.first, index % dims.second);
		}
		
		void operator()(int col, int row){
			operator()(col, row, col+1, row+1);
		}
		
		void operator()(int start_c, int start_r, int end_c, int end_r){
			this->sub_matrix = true;
			this->sub_col_start = start_c;
			this->sub_row_start = start_r;
			this->sub_col_end = end_c;
			this->sub_row_end = end_r;
		}
		
		//Overridden for operators like multiplication
		virtual void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(start_c, start_r, end_c, end_r, has_req);
			}
		}
		
		//Gets the size of an expression result without evaluating
		virtual size_t get_size() {
			if (!this->dims_cached) {
				this->get_dims();
			}
			
			return this->cached_size;
		}
		
		//Gets the dimensions of an expression without evaluating
		pair<size_t, size_t> get_dims() {
			if (!this->dims_cached){
				if (this->children.size() > 0){
					this->get_children_dims();
				} else {
					this->cached_dims = make_pair(this->matrix.cols(), this->matrix.rows());
					this->cached_size = this->matrix.size();
				}
			}
			
			return this->cached_dims;
		}
		
		//Overridden for operators like multiplication
		virtual void get_children_dims() {
			this->cached_dims = this->children[0]->get_dims();
			this->cached_size = this->children[0]->get_size();
		}
		
		//Gets if an full expression will be on the device
		virtual size_t get_device() {
			if (this->children.size() > 0){
				bool result = true;
				for(int i = 0; i < this->children.size(); ++i){
					result = result && this->children[i]->get_device();
				}
				
				return result;
			}
			
			return this->matrix.device();
		}
		
		//Overriden by subclasses to evaluate an operator
		virtual void eval_internal(LaspMatrix<T>& result) {}
		
		//Store matrix and constant arguments to function
		void add_arguments() {
			for (int i = 0; i < this->children.size(); ++i){
				shared_ptr<EvalBase<T> > child = this->children[i];
				child->parent = this;
			}
		}
		
		template<class A, class ... ARGS>
		void add_arguments(A arg, ARGS ... args){
			add_argument(arg);
			add_arguments(args...);
		}
		
		void add_argument(EvalBase<T> arg){
			this->children.push_back(shared_ptr<EvalBase<T> > (new EvalBase<T> (arg)));
		}
		
		void add_argument(LaspMatrix<T> arg){
			this->children.push_back(shared_ptr<EvalBase<T> > (new EvalBase<T> (arg)));
		}
		
		void add_argument(shared_ptr<EvalBase<T> > arg){
			this->children.push_back(arg);
		}
		
		void add_argument(T arg){
			this->constants.push_back(arg);
		}
		
		//Preprocess the expression tree
		void preprocess() {
			for (int i = 0; i < this->children.size(); ++i){
				this->children[i] = this->children[i]->preprocess(this->children[i]);
			}
		}
		
		virtual shared_ptr<EvalBase<T> > preprocess(shared_ptr<EvalBase<T> > myself){
			this->preprocess();
			return myself;
		}
		
		//Find the expression subtree that can make best use of the preallocated output matrix
		virtual int get_pass_ind() {
			int pass_ind = -1;
			size_t best_size = 0;
			for (int i = 0; i < this->children.size(); ++i) {
				shared_ptr<EvalBase<T> > child = this->children[i];
				size_t child_size = child->get_size();
				if (child->get_device() == this->matrix.device() && this->matrix.mSize() >= child_size
						&& child_size > best_size && (!this->matrix.isSubMatrix() || this->matrix.size() == child_size)) {
					pass_ind = i;
					best_size = child_size;
				}
			}
			
			return pass_ind;
		}
		
		//Evaluate the expression rooted at this node
		LaspMatrix<T> eval() {
			if (this->has_matrix) {
				return this->matrix;
			}
			
			pair<size_t, size_t> dims = this->get_dims();
			this->matrix.resize(dims.first, dims.second);

			int pass_ind = this->get_pass_ind();
			for (int i = 0; i < this->children.size(); ++i) {
				shared_ptr<EvalBase<T> > child = this->children[i];
				if (i == pass_ind) {
					this->child_mats.push_back(child->eval(this->matrix));
				}
				else {
					LaspMatrix<T> output;
					this->child_mats.push_back(child->eval(output));
				}
			}
			
			this->eval_internal(this->matrix);
			this->full_matrix = this->matrix;
			this->has_matrix = true;
			
			return this->matrix;
		}
		
		//Evaluate using a preallocated result matrix
		LaspMatrix<T> eval(LaspMatrix<T>& result){
			if (this->has_matrix) {
				return this->matrix;
			}
			
			this->matrix = result;
			return this->eval();
		}
		
		//Check that we won't be overwriting an input that is used more than once
		int check_key(LaspMatrix<T>& result) {
			int match = 0;
			match += this->matrix.key() == result.key() ? 1 : 0;
			for (int i = 0; i < this->children.size(); ++i){
				match += this->children[i]->check_key(result);
			}
			
			return match;
		}
		
		//Execute the expression tree and store the result in the input matrix
		void evaluate_full(LaspMatrix<T>& result){
			this->set_submatrix();
			this->preprocess();
			if (this->check_key(result) > 1){
				LaspMatrix<T> new_result;
				this->eval(new_result);
				result = new_result;
			}
			else {
				this->eval(result);
			}
			
			result.operator=(this->matrix);
		}
		
		//Implicit conversion to a matrix forces the expression tree to execute
		operator LaspMatrix<T>(){
			this->set_submatrix();
			this->preprocess();
			return this->eval();
		}
		
		
		virtual shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			if (this->children.size() == 1) {
				return this->children[0]->deriv(target, result_size);
			}
			
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			return shared_ptr<EvalBase<T> >(0);
		}
		
		shared_ptr<EvalBase<T> > deriv(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			if (this->sub_matrix) {
#ifndef NDEBUG
				cerr << "Cannot take derivative with respect to a sub-matrix" << endl;
#endif
				return shared_ptr<EvalBase<T> >(0);
			}
			
			if(this->is_matrix) {
				if (this->matrix.key() == target.key()) {
					return shared_ptr<EvalBase<T> > (new EvalBase<T>(LaspMatrix<T>::eye(target.rows())));
				}
				else {
					return shared_ptr<EvalBase<T> > (new EvalBase<T>(LaspMatrix<T>::zeros(result_size.first, result_size.second)));
				}
			}
			
			if (this->check_key(target) == 0) {
				return shared_ptr<EvalBase<T> > (new EvalBase<T>(LaspMatrix<T>::zeros(result_size.first, result_size.second)));
			}
			
			return this->deriv_internal(target, result_size);
		}
		
		shared_ptr<EvalBase<T> > deriv_full(LaspMatrix<T> target){
			this->set_submatrix();
			
			pair<size_t, size_t> my_size = this->get_dims();
			
			if (target.cols() > 1 || my_size.first > 1) {
#ifndef NDEBUG
				cerr << "Derivative targets must be column vectors or scalars" << endl;
#endif
				return shared_ptr<EvalBase<T> > (0);
			}
			
			pair<size_t, size_t> result_size = make_pair(target.rows(), my_size.second);
			return this->deriv(target, result_size);
		}
		
	};
	
	//Copying into a matrix call evaluation
	template<class T>
	template<class N>
	LaspMatrix<T>::LaspMatrix(const shared_ptr<N >& other) : LaspMatrix<T>() {
		other->evaluate_full(*this);
	}
	
	
	//Sets matrix equal to the result of an expression
	template<class T>
	template<class N>
	LaspMatrix<T>& LaspMatrix<T>::operator=(const shared_ptr<N >& other) {
		//Allow the expression to use our memory if we're the only reference to it
		if(rc() == 1) {
			other->evaluate_full(*this);
		} else {
			LaspMatrix<T> new_data;
			other->evaluate_full(new_data);
			operator=(new_data);
		}
		
		return *this;
	}
	
	//Get expression derivative
	template <class T>
	shared_ptr<EvalBase<T> > der(EvalBase<T> arg1, LaspMatrix<T> target) {
		return arg1.deriv_full(target);
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > der(LaspMatrix<T> arg1, LaspMatrix<T> target) {
		shared_ptr<EvalBase<T> > ret_val = shared_ptr<EvalBase<T> >(new EvalBase<T>(arg1));
		return ret_val->deriv_full(target);
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > der(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> target) {
		return arg1->deriv_full(target);
	}
	
	//Print a matrix
	template <class T>
	LaspMatrix<T> print(EvalBase<T> arg1, string name = "") {
		LaspMatrix<T> output;
		arg1.evaluate_full(output);
		output.printMatrix(name);
		return output;
	}
	
	template <class T>
	LaspMatrix<T> print(LaspMatrix<T> arg1, string name = "") {
		arg1.printMatrix(name);
		return arg1;
	}
	
	template <class T>
	LaspMatrix<T> print(shared_ptr<EvalBase<T> > arg1, string name = "") {
		LaspMatrix<T> output;
		arg1->evaluate_full(output);
		output.printMatrix(name);
		return output;
	}
	
	template <class T>
	LaspMatrix<T> info(EvalBase<T> arg1, string name = "") {
		LaspMatrix<T> output;
		arg1.evaluate_full(output);
		output.printInfo(name);
		return output;
	}
	
	template <class T>
	LaspMatrix<T> info(LaspMatrix<T> arg1, string name = "") {
		arg1.printInfo(name);
		return arg1;
	}
	
	template <class T>
	LaspMatrix<T> info(shared_ptr<EvalBase<T> > arg1, string name = "") {
		LaspMatrix<T> output;
		arg1->evaluate_full(output);
		output.printInfo(name);
		return output;
	}
	
	//Copy a matrix
	template <class T>
	LaspMatrix<T> copy(EvalBase<T> arg1) {
		LaspMatrix<T> output;
		arg1.evaluate_full(output);
		return output;
	}
	
	template <class T>
	LaspMatrix<T> copy(LaspMatrix<T> arg1) {
		return arg1.copy();
	}
	
	template <class T>
	LaspMatrix<T> copy(shared_ptr<EvalBase<T> > arg1) {
		LaspMatrix<T> output;
		arg1->evaluate_full(output);
		return output;
	}
	
	//Subselection operators
	template<class T>
	shared_ptr<EvalBase<T> > sel(shared_ptr<EvalBase<T> > mat, int arg1) {
		mat->operator()(arg1);
		return mat;
	}
	
	template<class T>
	shared_ptr<EvalBase<T> > sel(shared_ptr<EvalBase<T> > mat, int arg1, int arg2) {
		mat->operator()(arg1, arg2);
		return mat;
	}
	
	template<class T>
	shared_ptr<EvalBase<T> > sel(shared_ptr<EvalBase<T> > mat, int arg1, int arg2, int arg3, int arg4) {
		mat->operator()(arg1, arg2, arg3, arg4);
		return mat;
	}
	
	template<class T>
	EvalBase<T> sel(EvalBase<T> mat, int arg1) {
		mat.operator()(arg1);
		return mat;
	}
	
	template<class T>
	EvalBase<T> sel(EvalBase<T> mat, int arg1, int arg2) {
		mat.operator()(arg1, arg2);
		return mat;
	}
	
	template<class T>
	EvalBase<T> sel(EvalBase<T> mat, int arg1, int arg2, int arg3, int arg4) {
		mat.operator()(arg1, arg2, arg3, arg4);
		return mat;
	}
	
	template<class T>
	LaspMatrix<T> sel(LaspMatrix<T> mat, int arg1) {
		return mat.operator()(arg1);
	}
	
	template<class T>
	LaspMatrix<T> sel(LaspMatrix<T> mat, int arg1, int arg2) {
		return mat.operator()(arg1, arg2);
	}
	
	template<class T>
	LaspMatrix<T> sel(LaspMatrix<T> mat, int arg1, int arg2, int arg3, int arg4) {
		return mat.operator()(arg1, arg2, arg3, arg4);
	}
	
	template<class T>
	shared_ptr<EvalBase<T> > csel(shared_ptr<EvalBase<T> > mat, int arg1) {
		mat->operator()(arg1, 0, arg1 + 1, -1);
		return mat;
	}
	
	template<class T>
	shared_ptr<EvalBase<T> > csel(shared_ptr<EvalBase<T> > mat, int arg1, int arg2) {
		mat->operator()(arg1, 0, arg2, -1);
		return mat;
	}
	
	template<class T>
	EvalBase<T> csel(EvalBase<T> mat, int arg1) {
		mat.operator()(arg1, 0, arg1 + 1, -1);
		return mat;
	}
	
	template<class T>
	EvalBase<T> csel(EvalBase<T> mat, int arg1, int arg2) {
		mat.operator()(arg1, 0, arg2, -1);
		return mat;
	}
		
	template<class T>
	LaspMatrix<T> csel(LaspMatrix<T> mat, int arg1) {
		return mat.operator()(arg1, 0, arg1 + 1, mat.rows());
	}
	
	template<class T>
	LaspMatrix<T> csel(LaspMatrix<T> mat, int arg1, int arg2) {
		return mat.operator()(arg1, 0, arg2, mat.rows());
	}
	
	template<class T>
	shared_ptr<EvalBase<T> > rsel(shared_ptr<EvalBase<T> > mat, int arg1) {
		mat->operator()(0, arg1, -1, arg1 + 1);
		return mat;
	}
	
	template<class T>
	shared_ptr<EvalBase<T> > rsel(shared_ptr<EvalBase<T> > mat, int arg1, int arg2) {
		mat->operator()(0, arg1, -1, arg2);
		return mat;
	}
	
	template<class T>
	EvalBase<T> rsel(EvalBase<T> mat, int arg1) {
		mat.operator()(0, arg1, -1, arg1 + 1);
		return mat;
	}
	
	template<class T>
	EvalBase<T> rsel(EvalBase<T> mat, int arg1, int arg2) {
		mat.operator()(0, arg1, -1, arg2);
		return mat;
	}
	
	template<class T>
	LaspMatrix<T> rsel(LaspMatrix<T> mat, int arg1) {
		return mat.operator()(0, arg1, mat.rows(), arg1 + 1);
	}
	
	template<class T>
	LaspMatrix<T> rsel(LaspMatrix<T> mat, int arg1, int arg2) {
		return mat.operator()(0, arg1, mat.rows(), arg2);
	}

	template<class T>
	struct addEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		addEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			if (this->constants.size() > 0){
				return this->children[0]->deriv(target, result_size);
			} else {
				if (this->children[0]->check_key(target) == 0) {
					return this->children[1]->deriv(target, result_size);
				}
				
				if (this->children[1]->check_key(target) == 0) {
					return this->children[0]->deriv(target, result_size);
				}
				
				return this->children[0]->deriv(target, result_size) + this->children[1]->deriv(target, result_size);
			}
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			if (this->constants.size() > 0){
				this->child_mats[0].add(this->constants[0], result);
			} else {
				this->child_mats[0].add(this->child_mats[1], result);
			}
		}
	};
	
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator+(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator+(N arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(EvalBase<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator+(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator+(N arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(LaspMatrix<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(EvalBase<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(LaspMatrix<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator+(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator+(N arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(shared_ptr<EvalBase<T> > arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(LaspMatrix<T> arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new addEvaluator<T>(arg1, arg2));
	}
	
	template<class T>
	struct subEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		subEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			if (this->constants.size() > 0){
				return this->children[0]->deriv(target, result_size);
			} else {
				if (this->children[0]->check_key(target) == 0) {
					return -(this->children[1]->deriv(target, result_size));
				}
				
				if (this->children[1]->check_key(target) == 0) {
					return this->children[0]->deriv(target, result_size);
				}
				
				return this->children[0]->deriv(target, result_size) - this->children[1]->deriv(target, result_size);
			}
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			if (this->constants.size() > 0){
				this->child_mats[0].subtract(this->constants[0], result);
			} else {
				this->child_mats[0].subtract(this->child_mats[1], result);
			}
		}
	};
	
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator-(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator-(N arg1, EvalBase<T> arg2) {
		return arg1 + (-arg2);
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(EvalBase<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator-(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator-(N arg1, LaspMatrix<T> arg2) {
		return arg1 + (-arg2);
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(LaspMatrix<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(EvalBase<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(LaspMatrix<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator-(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator-(N arg1, shared_ptr<EvalBase<T> > arg2) {
		return arg1 + (-arg2);
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(shared_ptr<EvalBase<T> > arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(LaspMatrix<T> arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new subEvaluator<T>(arg1, arg2));
	}
	
	template<class T>
	struct negEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		negEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			return -(this->children[0]->deriv(target, result_size));
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].negate(result);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new negEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new negEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator-(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new negEvaluator<T>(arg1));
	}
	
	template<class T>
	struct posEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		posEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			return this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			result.copy(this->child_mats[0]);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new posEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new posEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator+(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new posEvaluator<T>(arg1));
	}
	
	template<class T>
	struct logEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		logEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			
			if (this->children[0]->get_dims().first == 1) {
				return mul((1 / this->children[0]) * temp_der, LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
			}
			
			return mul(t(temp_der) * (1 / this->children[0]), LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].log(result);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > log(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new logEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > log(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new logEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > log(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new logEvaluator<T>(arg1));
	}
	
	template<class T>
	struct expEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		expEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			
			if (this->children[0]->get_dims().first == 1) {
				return mul(exp(this->children[0]) * temp_der, LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
			}
			
			return mul(t(temp_der) * exp(this->children[0]), LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].exp(result);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > exp(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new expEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > exp(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new expEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > exp(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new expEvaluator<T>(arg1));
	}
	
	template<class T>
	struct tanhEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		tanhEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			
			if (this->children[0]->get_dims().first == 1) {
				return mul((1 - pow(tanh(this->children[0]), 2)) * temp_der, LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
			}
			
			return mul(t(temp_der) * (1 - pow(tanh(this->children[0]), 2)), LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].tanh(result);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > tanh(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new tanhEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > tanh(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new tanhEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > tanh(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new tanhEvaluator<T>(arg1));
	}
	
	template<class T>
	struct powEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		powEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			
			if (this->children[0]->get_dims().first == 1) {
				return mul((this->constants[0] * pow(this->children[0], this->constants[0] - 1)) * temp_der, LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
			}
			
			return mul(t(temp_der) * (this->constants[0] * pow(this->children[0], this->constants[0] - 1)), LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].pow(this->constants[0], result);
		}
	};
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > pow(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new powEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > pow(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new powEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > pow(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new powEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > sqrt(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new powEvaluator<T>(arg1, 0.5));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > sqrt(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new powEvaluator<T>(arg1, 0.5));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > sqrt(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new powEvaluator<T>(arg1, 0.5));
	}
	
	template<class T>
	struct pdfEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		pdfEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			
			if (this->children[0]->get_dims().first == 1) {
				return mul(((-this->children[0] / (this->constants[1] * this->constants[1])) * pdf(this->children[0], this->constants[0], this->constants[1])) * temp_der, LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
			}
			
			return mul(t(temp_der) * ((-this->children[0] / (this->constants[1] * this->constants[1])) * pdf(this->children[0], this->constants[0], this->constants[1])), LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			if (this->constants.size() == 0) {
				this->child_mats[0].normPDF(result);
			} else if (this->constants.size() == 1) {
				this->child_mats[0].normPDF(result, this->constants[0]);
			} else {
				this->child_mats[0].normPDF(result, this->constants[0], this->constants[1]);
			}
		}
	};
	
	
	template <class T, class N = int, class M = int>
	shared_ptr<EvalBase<T> > pdf(EvalBase<T> arg1, N arg2 = 0, M arg3 = 1) {
		return shared_ptr<EvalBase<T> >(new pdfEvaluator<T>(arg1, arg2, arg3));
	}
	
	template <class T, class N = int, class M = int>
	shared_ptr<EvalBase<T> > pdf(LaspMatrix<T> arg1, N arg2 = 0, M arg3 = 1) {
		return shared_ptr<EvalBase<T> >(new pdfEvaluator<T>(arg1, arg2, arg3));
	}
	
	template <class T, class N = int, class M = int>
	shared_ptr<EvalBase<T> > pdf(shared_ptr<EvalBase<T> > arg1, N arg2 = 0, M arg3 = 1) {
		return shared_ptr<EvalBase<T> >(new pdfEvaluator<T>(arg1, arg2, arg3));
	}
	
	template<class T>
	struct cdfEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		cdfEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			
			if (this->children[0]->get_dims().first == 1) {
				return mul((pdf(this->children[0], this->constants[0], this->constants[1])) * temp_der, LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
			}
			
			return mul(t(temp_der) * (pdf(this->children[0], this->constants[0], this->constants[1])), LaspMatrix<T>::eye(this->children[0]->get_size())) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			if (this->constants.size() == 0) {
				this->child_mats[0].normCDF(result);
			} else if (this->constants.size() == 1) {
				this->child_mats[0].normCDF(result, this->constants[0]);
			} else {
				this->child_mats[0].normCDF(result, this->constants[0], this->constants[1]);
			}
		}
	};
	
	
	template <class T, class N = int, class M = int>
	shared_ptr<EvalBase<T> > cdf(EvalBase<T> arg1, N arg2 = 0, M arg3 = 1) {
		return shared_ptr<EvalBase<T> >(new cdfEvaluator<T>(arg1, arg2, arg3));
	}
	
	template <class T, class N = int, class M = int>
	shared_ptr<EvalBase<T> > cdf(LaspMatrix<T> arg1, N arg2 = 0, M arg3 = 1) {
		return shared_ptr<EvalBase<T> >(new cdfEvaluator<T>(arg1, arg2, arg3));
	}
	
	template <class T, class N = int, class M = int>
	shared_ptr<EvalBase<T> > cdf(shared_ptr<EvalBase<T> > arg1, N arg2 = 0, M arg3 = 1) {
		return shared_ptr<EvalBase<T> >(new cdfEvaluator<T>(arg1, arg2, arg3));
	}
	
	template<class T>
	struct transEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		transEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		//??????
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			return t(this->children[0]->deriv(target, result_size));
		}
		
		oper_type type() {
			return TRANSPOSE;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(start_r, start_c, end_r, end_c, has_req);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> dims = this->children[0]->get_dims();
			this->cached_dims = make_pair(dims.second, dims.first);
			this->cached_size = this->children[0]->get_size();
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			if (result.key() == this->child_mats[0].key()) {
				LaspMatrix<T> new_result;
				this->child_mats[0].transpose(new_result);
				result = new_result;
			}
			else {
				this->child_mats[0].transpose(result);
			}
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > t(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new transEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > t(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new transEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > t(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new transEvaluator<T>(arg1));
	}
	
	//Matrix multiplication
	template<class T>
	struct multEvaluator : public EvalBase<T> {
		bool trans1;
		bool trans2;
		
		template<class ... ARGS>
		multEvaluator(ARGS ... args) : trans1(false), trans2(false) {
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			//A*u(x) -> A*du/dx
			if (this->children[0]->check_key(target) == 0 && this->children[1]->check_key(target) > 0) {
				return this->children[0] * this->children[1]->deriv(target, result_size);
			}
			
			//u(x)' * A -> du/dx*A' ??????
			if (this->children[0]->check_key(target) > 0 && this->children[1]->check_key(target) == 0) {
				return this->children[0]->deriv(target, result_size) * t(this->children[1]);
			}
			
			//From here on, both sides are functions of x
			
			//a*u(x) -> da/dx + du/dx ?????
			if (this->children[0]->get_size() == 1 || this->children[1]->get_size() == 1) {
				return this->children[0] * this->children[1]->deriv(target, result_size) + this->children[1] * this->children[0]->deriv(target, result_size);
			}
			

			if (this->children[0]->get_dims().second == 1 && this->children[1]->get_dims().first == 1) {
				return this->children[0] * this->children[1]->deriv(target, result_size) + t(this->children[1]) * this->children[0]->deriv(target, result_size);
			}
			
			//Those are all of the identities listed on wikipedia, past here who the fuck knows...
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			return shared_ptr<EvalBase<T> >(0);

		}
		
		oper_type type() {
			return MULT;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			if (!has_req) {
				for(int i = 0; i < this->children.size(); ++i){
					this->children[i]->set_submatrix(start_r, start_c, end_r, end_c, has_req);
				}
				
				return;
			}
			
			if (trans1) {
				this->children[0]->set_submatrix(start_r, 0, end_r, -1, has_req);
			}
			else {
				this->children[0]->set_submatrix(0, start_r, -1, end_r, has_req);
			}
			
			if (trans2) {
				this->children[1]->set_submatrix(0, start_c, -1, end_c, has_req);
			}
			else {
				this->children[1]->set_submatrix(start_c, 0, end_c, -1, has_req);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> child1_dims = this->children[0]->get_dims();
			pair<size_t, size_t> child2_dims = this->children[1]->get_dims();
			
			size_t cols = trans2 ? child2_dims.second : child2_dims.first;
			size_t rows = trans1 ? child1_dims.first : child1_dims.second;
			
			this->cached_dims = make_pair(cols, rows);
			this->cached_size = cols * rows;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].multiply(this->child_mats[1], result, trans1, trans2);
		}
	};
	
	//Multiplication with constant
	template<class T>
	struct cmultEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		cmultEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			return this->constants[0] * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].multiply(this->constants[0], result);
		}
	};
	
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator*(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator*(N arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(EvalBase<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator*(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator*(N arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(LaspMatrix<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(EvalBase<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(LaspMatrix<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator*(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator*(N arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(shared_ptr<EvalBase<T> > arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator*(LaspMatrix<T> arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new multEvaluator<T>(arg1, arg2));
	}
	
	//Matrix division (system of equations)
	template<class T>
	struct solveEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		solveEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			//Pretty sure this won't work
			if (this->children[0]->check_key(target) > 0) {
#ifndef NDEBUG
				cerr << "Cannot take derivate of expression" << endl;
#endif
				return shared_ptr<EvalBase<T> >(0);
			}
			
			return inv(this->children[0]) * this->children[0]->deriv(target, result_size);
		}
		
		oper_type type() {
			return SOLVE;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(-1, -1, -1, -1, false);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> dims = this->children[1]->get_dims();
			
			int cols = this->end_col == -1 ? dims.first : this->end_col;
			cols -= this->start_col == -1 ? 0 : this->start_col;
			int rows = this->end_row == -1 ? dims.second : this->end_row;
			rows -= this->start_row == -1 ? 0 : this->start_row;
			
			this->cached_dims = make_pair(cols, rows);
			this->cached_size = cols * rows;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			LaspMatrix<T> output;
			
			if (this->start_col == -1 || this->start_row == -1) {
				this->child_mats[0].solve(this->child_mats[1], output);
				result.operator=(output);
				return;
			}
			
			this->child_mats[0].solve(this->child_mats[1], output);
			result.operator=(output(this->start_col, this->start_row, this->end_col == -1 ? output.cols() : this->end_col, this->end_row == -1 ? output.rows() : this->end_row).copy());
		}
	};
	
	//Division with constant
	template<class T>
	struct cdivEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		cdivEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			return (1.0 / this->constants[0]) * this->children[0]->deriv(target, result_size);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].multiply(1.0 / this->constants[0], result);
		}
	};
	
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator/(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cdivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator/(N arg1, EvalBase<T> arg2) {
		return pow(arg2, -1) * arg1;
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(EvalBase<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator/(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cdivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator/(N arg1, LaspMatrix<T> arg2) {
		return pow(arg2, -1) * arg1;
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(LaspMatrix<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(EvalBase<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(LaspMatrix<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator/(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cdivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > operator/(N arg1, shared_ptr<EvalBase<T> > arg2) {
		return pow(arg2, -1) * arg1;
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(shared_ptr<EvalBase<T> > arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > operator/(LaspMatrix<T> arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new solveEvaluator<T>(arg1, arg2));
	}
	
	//Element-wise multiplication
	template<class T>
	struct emultEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		emultEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			LaspMatrix<T> temp_der1 = LaspMatrix<T>::ones(this->children[0]->get_size(), 1);
			LaspMatrix<T> temp_der2 = LaspMatrix<T>::ones(this->children[1]->get_size(), 1);
			
			shared_ptr<EvalBase<T> > arg1, arg2;
			if (this->children[0]->get_dims().first == 1) {
				arg1 = mul(this->children[0] * temp_der1, LaspMatrix<T>::eye(this->children[0]->get_size()));
			} else {
				arg1 = mul(t(temp_der1) * this->children[0], LaspMatrix<T>::eye(this->children[0]->get_size()));
			}
			
			if (this->children[1]->get_dims().first == 1) {
				arg2 = mul(this->children[0] * temp_der2, LaspMatrix<T>::eye(this->children[0]->get_size()));
			} else {
				arg2 = mul(t(temp_der2) * this->children[0], LaspMatrix<T>::eye(this->children[0]->get_size()));
			}
			
			if (this->children[0]->check_key(target) == 0) {
				return mul(arg1, this->children[1]->deriv(target, result_size));
			}
			
			if (this->children[1]->check_key(target) == 0) {
				return mul(arg2, this->children[0]->deriv(target, result_size));
			}
			
			return mul(arg1, this->children[1]->deriv(target, result_size)) + mul(arg2, this->children[0]->deriv(target, result_size));
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].eWiseMultM(this->child_mats[1], result);
		}
	};
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > mul(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > mul(N arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(EvalBase<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > mul(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > mul(N arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(LaspMatrix<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(EvalBase<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(LaspMatrix<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > mul(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > mul(N arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new cmultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(shared_ptr<EvalBase<T> > arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > mul(LaspMatrix<T> arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new emultEvaluator<T>(arg1, arg2));
	}
	
	//Element-wise division
	template<class T>
	struct edivEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		edivEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].eWiseDivM(this->child_mats[1], result);
		}
	};
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > div(EvalBase<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cdivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > div(N arg1, EvalBase<T> arg2) {
		return pow(arg2, -1) * arg1;
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(EvalBase<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > div(LaspMatrix<T> arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cdivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > div(N arg1, LaspMatrix<T> arg2) {
		return pow(arg2, -1) * arg1;
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(LaspMatrix<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(EvalBase<T> arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(LaspMatrix<T> arg1, EvalBase<T> arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > div(shared_ptr<EvalBase<T> > arg1, N arg2) {
		return shared_ptr<EvalBase<T> >(new cdivEvaluator<T>(arg1, arg2));
	}
	
	template <class T, class N>
	shared_ptr<EvalBase<T> > div(N arg1, shared_ptr<EvalBase<T> > arg2) {
		return pow(arg2, -1) * arg1;
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(shared_ptr<EvalBase<T> > arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(shared_ptr<EvalBase<T> > arg1, LaspMatrix<T> arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > div(LaspMatrix<T> arg1, shared_ptr<EvalBase<T> > arg2) {
		return shared_ptr<EvalBase<T> >(new edivEvaluator<T>(arg1, arg2));
	}
	
	template<class T>
	struct maxEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		maxEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			return shared_ptr<EvalBase<T> >(0);
		}
		
		oper_type type() {
			return VAL;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(-1, -1, -1, -1, false);
			}
		}
		
		void get_children_dims() {
			this->cached_dims = make_pair((size_t) 1, (size_t) 1);
			this->cached_size = (size_t) 1;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			result = (this->child_mats[0].maxElem());
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > max(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new maxEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > max(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new maxEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > max(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new maxEvaluator<T>(arg1));
	}
	
	template<class T>
	struct minEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		minEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			
			return shared_ptr<EvalBase<T> >(0);
		}
		
		oper_type type() {
			return VAL;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(-1, -1, -1, -1, false);
			}
		}
		
		void get_children_dims() {
			this->cached_dims = make_pair((size_t) 1, (size_t) 1);
			this->cached_size = (size_t) 1;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			result = (this->child_mats[0].minElem());
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > min(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new minEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > min(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new minEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > min(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new minEvaluator<T>(arg1));
	}
	
	template<class T>
	struct traceEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		traceEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			return shared_ptr<EvalBase<T> >(0);
		}
		
		oper_type type() {
			return VAL;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(-1, -1, -1, -1, false);
			}
		}
		
		void get_children_dims() {
			this->cached_dims = make_pair((size_t) 1, (size_t) 1);
			this->cached_size = (size_t) 1;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			result = (this->child_mats[0].trace());
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > trace(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new traceEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > trace(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new traceEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > trace(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new traceEvaluator<T>(arg1));
	}
	
	template<class T>
	struct diagEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		diagEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			return shared_ptr<EvalBase<T> >(0);
		}
		
		oper_type type() {
			return VEC;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			if (!has_req) {
				this->children[0]->set_submatrix(-1, -1, -1, -1, has_req);
			}
			else {
				this->children[0]->set_submatrix(start_r, start_r, end_r, end_r, has_req);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> dims = this->children[0]->get_dims();
			size_t dim = std::min(dims.first, dims.second);
			
			this->cached_dims = make_pair(1, dim);
			this->cached_size = (size_t) dim;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			result.operator=(this->child_mats[0].diag().copy());
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > diag(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new diagEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > diag(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new diagEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > diag(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new diagEvaluator<T>(arg1));
	}
	
	template<class T>
	struct csumEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		csumEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
			shared_ptr<EvalBase<T> > temp = this->children[0]->deriv(target, result_size);
			return LaspMatrix<T>::ones(temp->get_dims().second, 1) * temp;
		}
		
		oper_type type() {
			return VEC;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			if (!has_req) {
				this->children[0]->set_submatrix(-1, -1, -1, -1, has_req);
			}
			else {
				this->children[0]->set_submatrix(start_c, 0, end_c, -1, has_req);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> dims = this->children[0]->get_dims();
			size_t dim = dims.first;
			
			this->cached_dims = make_pair(dim, 1);
			this->cached_size = (size_t) dim;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].colSum(result);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > csum(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new csumEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > csum(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new csumEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > csum(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new csumEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > rsum(EvalBase<T> arg1) {
		return t(csum(t(arg1)));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > rsum(LaspMatrix<T> arg1) {
		return t(csum(t(arg1)));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > rsum(shared_ptr<EvalBase<T> > arg1) {
		return t(csum(t(arg1)));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > sum(EvalBase<T> arg1) {
		return rsum(csum(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > sum(LaspMatrix<T> arg1) {
		return rsum(csum(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > sum(shared_ptr<EvalBase<T> > arg1) {
		return rsum(csum(arg1));
	}
	
	template<class T>
	struct normEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		normEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		oper_type type() {
			return VEC;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			if (!has_req) {
				this->children[0]->set_submatrix(-1, -1, -1, -1, has_req);
			}
			else {
				this->children[0]->set_submatrix(start_c, 0, end_c, -1, has_req);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> dims = this->children[0]->get_dims();
			size_t dim = dims.first;
			
			this->cached_dims = make_pair(dim, 1);
			this->cached_size = (size_t) dim;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].colSqSum(result);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > norm(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new normEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > norm(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new normEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > norm(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new normEvaluator<T>(arg1));
	}
	
	template<class T>
	LaspMatrix<T> eye(size_t n){
		return LaspMatrix<T>::eye(n);
	}
	
	template<class T>
	LaspMatrix<T> zeros(size_t cols, size_t rows){
		return LaspMatrix<T>::ones(cols, rows);
	}
	
	template<class T>
	LaspMatrix<T> zeros(size_t n){
		return LaspMatrix<T>::zeros(n, n);
	}
	
	template<class T>
	LaspMatrix<T> ones(size_t cols, size_t rows){
		return LaspMatrix<T>::ones(cols, rows);
	}
	
	template<class T>
	LaspMatrix<T> ones(size_t n){
		return LaspMatrix<T>::ones(n, n);
	}
	
	template<class T>
	LaspMatrix<T> rand(size_t cols, size_t rows){
		return LaspMatrix<T>::random(cols, rows);
	}
	
	template<class T>
	LaspMatrix<T> rand(size_t n){
		return LaspMatrix<T>::random(n, n);
	}
	
	template<class T>
	LaspMatrix<T> vcat(LaspMatrix<T> top, LaspMatrix<T> bottom){
		return LaspMatrix<T>::vcat(top, bottom);
	}
	
	template<class ... ARGS, class T>
	LaspMatrix<T> vcat(LaspMatrix<T> top, ARGS ... args){
		return vcat(top, vcat(args...));
	}
	
	template<class T>
	LaspMatrix<T> hcat(LaspMatrix<T> left, LaspMatrix<T> right){
		return LaspMatrix<T>::hcat(left, right);
	}
	
	template<class ... ARGS, class T>
	LaspMatrix<T> hcat(LaspMatrix<T> top, ARGS ... args){
		return hcat(top, hcat(args...));
	}
	
	template <class T, class N>
	LaspMatrix<T>& operator+=(LaspMatrix<T>& mat, N arg1) {
		return mat = mat + arg1;
	}
	
	template <class T, class N>
	LaspMatrix<T>& operator-=(LaspMatrix<T>& mat, N arg1) {
		return mat = mat - arg1;
	}

	template <class T, class N>
	LaspMatrix<T>& operator*=(LaspMatrix<T>& mat, N arg1) {
		return mat = mat * arg1;
	}
	
	template <class T, class N>
	LaspMatrix<T>& operator/=(LaspMatrix<T>& mat, N arg1) {
		return mat = mat / arg1;
	}
	
	template <class T>
	LaspMatrix<T>& operator++(LaspMatrix<T>& mat) {
		return mat = mat + 1;
	}
	
	template <class T>
	LaspMatrix<T>& operator--(LaspMatrix<T>& mat) {
		return mat = mat - 1;
	}
	
	template<class T>
	struct invEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		invEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		shared_ptr<EvalBase<T> > deriv_internal(LaspMatrix<T> target, pair<size_t, size_t> result_size) {
#ifndef NDEBUG
			cerr << "Cannot take derivate of expression" << endl;
#endif
			return shared_ptr<EvalBase<T> >(0);
		}
		
		oper_type type() {
			return SOLVE;
		}
		
		void set_children_submatrices(int start_c, int start_r, int end_c, int end_r, bool has_req){
			for(int i = 0; i < this->children.size(); ++i){
				this->children[i]->set_submatrix(-1, -1, -1, -1, false);
			}
		}
		
		void get_children_dims() {
			pair<size_t, size_t> dims = this->children[0]->get_dims();
			
			int cols = this->end_col == -1 ? dims.first : this->end_col;
			cols -= this->start_col == -1 ? 0 : this->start_col;
			int rows = this->end_row == -1 ? dims.second : this->end_row;
			rows -= this->start_row == -1 ? 0 : this->start_row;
			
			this->cached_dims = make_pair(cols, rows);
			this->cached_size = cols * rows;
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			LaspMatrix<T> output;
			LaspMatrix<T> identity = eye(this->child_mats[0].cols());
			
			if (this->start_col == -1 || this->start_row == -1) {
				this->child_mats[0].solve(identity, output);
				result.operator=(output);
				return;
			}
			
			this->child_mats[0].solve(identity, output);
			result.operator=(output(this->start_col, this->start_row, this->end_col == -1 ? output.cols() : this->end_col, this->end_row == -1 ? output.rows() : this->end_row).copy());
		}

	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > inv(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new invEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > inv(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new invEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > inv(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new invEvaluator<T>(arg1));
	}
	
	template<class T>
	struct deviceEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		deviceEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].transferToDevice();
			result.operator=(this->child_mats[0]);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > device(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new deviceEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > device(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new deviceEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > device(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new deviceEvaluator<T>(arg1));
	}
	
	template<class T>
	struct hostEvaluator : public EvalBase<T> {
		template<class ... ARGS>
		hostEvaluator(ARGS ... args){
			this->add_arguments(args...);
		}
		
		void eval_internal(LaspMatrix<T>& result)  {
			this->child_mats[0].transferToHost();
			result.operator=(this->child_mats[0]);
		}
	};
	
	
	template <class T>
	shared_ptr<EvalBase<T> > host(EvalBase<T> arg1) {
		return shared_ptr<EvalBase<T> >(new hostEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > host(LaspMatrix<T> arg1) {
		return shared_ptr<EvalBase<T> >(new hostEvaluator<T>(arg1));
	}
	
	template <class T>
	shared_ptr<EvalBase<T> > host(shared_ptr<EvalBase<T> > arg1) {
		return shared_ptr<EvalBase<T> >(new hostEvaluator<T>(arg1));
	}
	
}

#endif
