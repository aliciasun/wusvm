//
//  base_model.h
//  SP_SVM
//
//  Created by Gabriel Hope on 9/26/14.
//
//

#ifndef __SP_SVM__base_model__
#define __SP_SVM__base_model__

#include <iostream>
#include "options.h"
#include "lasp_matrix.h"

namespace lasp {

//Base class for all all prediction models
//	T: Feature type (float or double)
//	N: Prediction type (int, float or double)
template<class T, class N = int>
class Model {
protected:
	opt options_;
	
public:
	//Get the options
	opt& get_options(){
		return options_;
	}
	
	//Get vectorized hyperparameters for optimization
	virtual LaspMatrix<T> get_hyp() = 0;
	virtual vector<string> get_hyp_labels() = 0;
	virtual int set_hyp(LaspMatrix<T> hyp) = 0;
	
	//Train the given data
	virtual int train(LaspMatrix<T> X, LaspMatrix<N> y) = 0;
	virtual int retrain(LaspMatrix<T> X, LaspMatrix<N> y) = 0;
	
	//Get class/regression predictions, probabilities or hyperplane distances
	virtual int predict(LaspMatrix<T> X, LaspMatrix<N>& output) = 0;
	virtual int confidence(LaspMatrix<T> X, LaspMatrix<T>& output) = 0;
	virtual int predict_confidence(LaspMatrix<T> X, LaspMatrix<N>& output_predictions, LaspMatrix<T>& output_confidence) = 0;
	virtual int distance(LaspMatrix<T> X, LaspMatrix<T>& output) = 0;
	
	//Get accuracy, log-loss or scaled distance
	virtual T score(LaspMatrix<N> y_pred, LaspMatrix<N> y_actual) = 0;
	virtual T loss(LaspMatrix<T> y_prob, LaspMatrix<N> y_actual) = 0;
	virtual T loss(LaspMatrix<N> y_pred, LaspMatrix<T> y_prob, LaspMatrix<N> y_actual) = 0;
	
	//Predict and get accuracy
	virtual T test(LaspMatrix<T> X, LaspMatrix<T> y) = 0;
	virtual T test_loss(LaspMatrix<T> X, LaspMatrix<T> y) = 0;
	
	//Construct with data
	//virtual BaseModel(LaspMatrix<T> X, LaspMatrix<N> y);
	//virtual BaseModel(LaspMatrix<T> X, LaspMatrix<N> y, opt options);
	
	//Set data and clear data without training
	//virtual int data(LaspMatrix<T> X, LaspMatrix<N> y);
	//virtual int clear_data();
	
	//Subsample the data (i.e. for hyperparameter optimization)
	//virtual int subsample(T rate);
	
	//Train with internal data
	//virtual int train();
	
	//Reset model
	//virtual int reset_model();
	
	//Holdout score on internal data
	//virtual T holdout_score(T holdout=0.1);
	//virtual T holdout_loss(T holdout=0.1);
	
	//Get holdout score/loss
	//virtual T operator(T holdout=0.1, bool loss=false);
	
	//virtual T operator(LaspMatrix<T> hyp, T holdout=0.1, bool loss=false);
};

}



#endif
