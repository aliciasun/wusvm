//
//  svm_model.h
//  SP_SVM
//
//  Created by Gabriel Hope on 9/26/14.
//
//

#ifndef __SP_SVM__svm_model__
#define __SP_SVM__svm_model__

#include <iostream>
#include "base_model.h"
#include "optimize.h"


namespace lasp {
	
	//Gaussian process regression model
	template<class T>
	class SVM: public Model<T, int> {
		LaspMatrix<T> Xin, Yin, Xs, alphas;
		vector<int> S;
		T bias;
		
		opt& options(){
			return this->options_;
		}
		
	public:
		SVM();
		SVM(opt opt_in);
		
		LaspMatrix<T> get_hyp();
		vector<string> get_hyp_labels();
		int set_hyp(LaspMatrix<T> hyp);
		
		int train(LaspMatrix<T> X, LaspMatrix<T> y);
		int train(LaspMatrix<T> X, LaspMatrix<T> y, bool optimize);
		int retrain(LaspMatrix<T> X, LaspMatrix<T> y);
		
		int predict(LaspMatrix<T> X, LaspMatrix<T>& output);
		int confidence(LaspMatrix<T> X, LaspMatrix<T>& output);
		int predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions, LaspMatrix<T>& output_confidence);
		int distance(LaspMatrix<T> X, LaspMatrix<T>& output);
		
		T score(LaspMatrix<T> y_pred, LaspMatrix<T> y_actual);
		T loss(LaspMatrix<T> y_prob, LaspMatrix<T> y_actual);
		T loss(LaspMatrix<T> y_pred, LaspMatrix<T> y_prob, LaspMatrix<T> y_actual);
		
		T test(LaspMatrix<T> X, LaspMatrix<T> y);
		T test_loss(LaspMatrix<T> X, LaspMatrix<T> y);
		
		vector<int> get_sv();
	};
	
	
	template<class T>
	SVM<T>::SVM() {
		options().kernel = LINEAR;
	}
	
	template<class T>
	SVM<T>::SVM(opt opt_in) {
		options() = opt_in;
	}
	
	template <class T>
	LaspMatrix<T> SVM<T>::get_hyp(){
		LaspMatrix<T> hyp(1, 2);
		hyp(0) = static_cast<T>(options().C);
		hyp(1) = static_cast<T>(options().eta);
		hyp(2) = static_cast<T>(options().set_size);
		
		switch (options().kernel) {
			case RBF:
				hyp.resize(1, 4);
				hyp(3) = static_cast<T>(options().gamma);
				break;
			case POLYNOMIAL:
				hyp.resize(1, 5);
				hyp(3) = static_cast<T>(options().coef);
				hyp(4) = static_cast<T>(options().degree);
				break;
			case SIGMOID:
				hyp.resize(1, 5);
				hyp(3) = static_cast<T>(options().coef);
				hyp(4) = static_cast<T>(options().gamma);
				break;
			default:
				break;
		}
		
		return hyp;
	}
	
	template <class T>
	vector<string> SVM<T>::get_hyp_labels(){
		vector<string> output;
		output.push_back("C");
		output.push_back("eta");
		output.push_back("set_size");
		
		switch (options().kernel) {
			case RBF:
				output.push_back("gamma");
				break;
			case POLYNOMIAL:
				output.push_back("coef");
				output.push_back("degree");
				break;
			case SIGMOID:
				output.push_back("coef");
				output.push_back("gamma");
				break;
			default:
				break;
		}
		
		return output;
	}
	
	template <class T>
	int SVM<T>::set_hyp(LaspMatrix<T> hyp){
		options().C = static_cast<double>(hyp(0));
		options().eta = static_cast<double>(hyp(1));
		options().set_size = static_cast<int>(hyp(2));
		
		switch (options().kernel) {
			case RBF:
				options().gamma = static_cast<double>(hyp(3));
				break;
			case POLYNOMIAL:
				options().coef = static_cast<double>(hyp(3));
				options().degree = static_cast<double>(hyp(4));
				break;
			case SIGMOID:
				options().coef = static_cast<double>(hyp(3));
				options().gamma = static_cast<double>(hyp(4));
				break;
			default:
				break;
		}
		
		return 0;
	}
	
	template <class T>
	int SVM<T>::train(LaspMatrix<T> X, LaspMatrix<T> y){
		Xin = LaspMatrix<T>::hcat(Xin, X);
		Yin = LaspMatrix<T>::hcat(Yin, y);
		
		
		return 0;
	}
	
	template <class T>
	int SVM<T>::retrain(LaspMatrix<T> X, LaspMatrix<T> y){
		Xin = LaspMatrix<T>();
		Yin = LaspMatrix<T>();
		return train(X, y);
	}
	
	template <class T>
	int SVM<T>::predict(LaspMatrix<T> X, LaspMatrix<T>& output){
		return -1;
	}
	
	template <class T>
	int SVM<T>::confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
		return -1;
	}
	
	template <class T>
	int SVM<T>::predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions , LaspMatrix<T>& output_confidence){
		
	}
	
	template <class T>
	int SVM<T>::distance(LaspMatrix<T> X, LaspMatrix<T>& output){
		return -1;
	}
	
	template <class T>
	T SVM<T>::score(LaspMatrix<T> y_pred, LaspMatrix<T> y_actual){
		return -1;
	}
	
	template <class T>
	T SVM<T>::loss(LaspMatrix<T> y_prob, LaspMatrix<T> y_actual){
		return -1;
	}
	
	template <class T>
	T SVM<T>::loss(LaspMatrix<T> y_pred, LaspMatrix<T> y_prob, LaspMatrix<T> y_actual){
		return -1;
	}
	
	template <class T>
	T SVM<T>::test(LaspMatrix<T> X, LaspMatrix<T> y){
		return -1;
	}
	
	template <class T>
	T SVM<T>::test_loss(LaspMatrix<T> X, LaspMatrix<T> y){
		return -1;
	}
	
	template<class T>
	vector<int> SVM<T>::get_sv() {
		return S;
	}
}


#endif /* defined(__SP_SVM__svm_model__) */
