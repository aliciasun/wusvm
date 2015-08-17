//
//  Header.h
//  SP_SVM
//
//  Created by Gabriel Hope on 10/11/14.
//
//

#ifndef SP_SVM_Header_h
#define SP_SVM_Header_h
#include "gp_model.h"

#define CPP11

#ifdef CPP11
#include <random>
#endif

namespace lasp {
	template<class T>
	class FStop {
		GP<T> model;
		GP<T> cost_model;
		
		LaspMatrix<T> iterations;
		LaspMatrix<T> costs;
		LaspMatrix<T> scores;
		
	public:
		FStop(int model_kernel = EXP, int cost_kernel = POLYNOMIAL);
		
		int train(LaspMatrix<T> new_iteration, LaspMatrix<T> new_score, bool optimize_gp = true);
		int train(LaspMatrix<T> new_iteration, LaspMatrix<T> new_score, LaspMatrix<T> new_cost, bool optimize_gp = true);
		int retrain(LaspMatrix<T> new_iterations, LaspMatrix<T> new_scores);
		int retrain(LaspMatrix<T> new_iterations, LaspMatrix<T> new_scores, LaspMatrix<T> new_costs);
		
		int predict(LaspMatrix<T> X, LaspMatrix<T>& output);
		int confidence(LaspMatrix<T> X, LaspMatrix<T>& output);
		int predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions, LaspMatrix<T>& output_confidence);
		
		int cost_predict(LaspMatrix<T> X, LaspMatrix<T>& output);
		int cost_confidence(LaspMatrix<T> X, LaspMatrix<T>& output);
		int cost_predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions, LaspMatrix<T>& output_confidence);
		
		T get_step(bool cost=true, int num_cand=300);
		T get_step(LaspMatrix<T> upper_bound, bool cost=true, int num_cand=300);
		
		T get_integer_step(T upper_bound, T error_thresh, T error_prob);
		
#ifdef CPP11
		T get_cost_sensitive_step(T upper_bound, T error_thresh, T error_prob);
#endif
		
		T stopping_value_gp();
		T stopping_value_heuristic();
		
	};


template<class T>
FStop<T>::FStop(int model_kernel, int cost_kernel){
	model.get_options().kernel = model_kernel;
	cost_model.get_options().kernel = cost_kernel;
}

template<class T>
int FStop<T>::train(LaspMatrix<T> new_iteration, LaspMatrix<T> new_score, bool optimize_gp) {
	return train(new_iteration, new_score, LaspMatrix<T>(), optimize_gp);
}

template<class T>
int FStop<T>::train(LaspMatrix<T> new_iteration, LaspMatrix<T> new_score, LaspMatrix<T> new_cost, bool optimize_gp) {
	iterations = LaspMatrix<T>::hcat(iterations, new_iteration);
	scores = LaspMatrix<T>::hcat(scores, new_score);
	costs = LaspMatrix<T>::hcat(costs, new_cost);
	
	model.train(new_iteration, new_score, optimize_gp);
	
	if (costs.size() == scores.size()){
		cost_model.train(new_iteration, new_cost, optimize_gp);
	}
	
	return 0;
}

template<class T>
int FStop<T>::retrain(LaspMatrix<T> new_iterations, LaspMatrix<T> new_scores) {
	return retrain(new_iterations, new_scores, LaspMatrix<T>());
}

template<class T>
int FStop<T>::retrain(LaspMatrix<T> new_iterations, LaspMatrix<T> new_scores, LaspMatrix<T> new_costs){
	iterations = LaspMatrix<T>();
	scores = LaspMatrix<T>();
	costs = LaspMatrix<T>();
	
	iterations = LaspMatrix<T>::hcat(iterations, new_iterations);
	scores = LaspMatrix<T>::hcat(scores, new_scores);
	costs = LaspMatrix<T>::hcat(costs, new_costs);
	
	model.retrain(iterations, scores);
	
	if (costs.size() == scores.size()){
		cost_model.retrain(iterations, costs);
	}
	
	return 0;
}

template<class T>
int FStop<T>::predict(LaspMatrix<T> X, LaspMatrix<T>& output){
	return model.predict(X, output);
}

template<class T>
int FStop<T>::confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
	return model.confidence(X, output);
}

template<class T>
int FStop<T>::predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions, LaspMatrix<T>& output_confidence){
	return model.predict_confidence(X, output_predictions, output_confidence);
}

template<class T>
int FStop<T>::cost_predict(LaspMatrix<T> X, LaspMatrix<T>& output){
	return cost_model.predict(X, output);
}

template<class T>
int FStop<T>::cost_confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
	return cost_model.confidence(X, output);
}

template<class T>
int FStop<T>::cost_predict_confidence(LaspMatrix<T> X, LaspMatrix<T>& output_predictions, LaspMatrix<T>& output_confidence){
	return cost_model.predict_confidence(X, output_predictions, output_confidence);
}

template<class T>
T FStop<T>::get_step(bool cost, int num_cand){
	return get_step(LaspMatrix<T>(), cost, num_cand);
}

template<class T>
T FStop<T>::get_step(LaspMatrix<T> upper_bound, bool cost, int num_cand){
	LaspMatrix<T> bound = upper_bound.copy();
	LaspMatrix<T> last_col = iterations(iterations.cols()-1, 0, iterations.cols(), iterations.rows());
	
	if (bound.size() == 0 && iterations.size() > 0) {
		last_col.multiply(10, bound);
	}
	
	LaspMatrix<T> test_points;
	set_parameter_grid_uniform(last_col, bound, test_points, num_cand);
	
	LaspMatrix<T> ei, new_costs, ei_out;
	
	model.expected_improvement(test_points, ei);
	if (cost && costs.size() == scores.size()){
		cost_model.predict(test_points, new_costs);
		ei.eWiseDivM(new_costs);
	}
	

	int row, best_col;
	T ei_value = ei.maxElem(best_col, row);
	LaspMatrix<T> ei_temp = test_points(best_col, 0, best_col+1, ei.rows());
	ei_out.copy(ei_temp);
	return ei_out(0,0);
}

	template<class T>
	T FStop<T>::get_integer_step(T upper_bound, T error_thresh, T error_prob){
		LaspMatrix<T> mean, sig;
		
		//Binary search variables
		int ubound = static_cast<int>(upper_bound);
		int lbound = static_cast<int>(iterations(iterations.size() - 1, 0)) + 1;
		int midpoint = lbound + ((ubound - lbound) / 2);
		
		T current_error = scores(scores.size() - 1, 0);
		LaspMatrix<T> target_error(1, 1, current_error - error_thresh);
		
		int min_point = ubound;
		
		while(ubound >= lbound) {
			T candidate = static_cast<T>(midpoint);
			LaspMatrix<T> cand_mat(1, 1, candidate);
			predict_confidence(cand_mat, mean, sig);
			LaspMatrix<T> candidate_prob(1,1);
			
			target_error.normCDF(candidate_prob, mean, sig);
			
			int will_stop = candidate_prob(0) < error_prob;
			
			if (!will_stop){
				ubound = midpoint - 1;
				min_point = std::min(midpoint, min_point);
			} else {
				lbound = midpoint + 1;
			}
			
			midpoint = lbound + ((ubound - lbound) / 2);
		}
		
		return static_cast<T>(min_point);
	}
	
	
#ifdef CPP11
	template<class T>
	T FStop<T>::get_cost_sensitive_step(T upper_bound, T error_thresh, T error_prob){
		//Hyperparameters
		int samples = 100;
		int num_steps = 25;
		
		//CPP11 random setup
		random_device rd;
		mt19937 gen(rd());
		
		//Get min step that meets our stopping criteria
		int M_best = get_integer_step(upper_bound, error_thresh, error_prob);
		T M_best_float = static_cast<T>(M_best);
		
		//Get the cost estimate for the M_best step
		LaspMatrix<T> cost_mean_mat, cost_sig_mat;
		cost_predict_confidence(LaspMatrix<T>(1,1,M_best_float), cost_mean_mat, cost_sig_mat);
		
		T M_best_cost = cost_mean_mat(0);
		T M_best_cost_sig = cost_sig_mat(0);
		
		//Declare variables for the best interior point
		T best_cost_val = numeric_limits<T>::max();
		T best_test = 0;
		
		//Search between our current point and the M_best step
		int lbound = static_cast<int>(iterations(iterations.size() - 1, 0)) + 1;
		
		int step_size = std::max(static_cast<int>((upper_bound - lbound) / num_steps), 1);
		for (int M_test = lbound; M_test < M_best; M_test += step_size) {
			
			//Estimated difference in cost b/w M_best and the interior test point
			T cost_val = 0;
			
			//Float of iteration to test
			T M_test_float = static_cast<T>(M_test);
			
			//Get the predicted error and cost for the test step
			LaspMatrix<T> M_test_mean, M_test_sig, M_test_cost;
			predict_confidence(LaspMatrix<T>(1,1,M_test_float), M_test_mean, M_test_sig);
			cost_predict(LaspMatrix<T>(1,1,M_test_float), M_test_cost);
			
			//Break if our test point is already half the cost of the best
			if (M_test_cost(0) > M_best_cost / 2.0) {
				break;
			}
			
			//Sample errors from the predicted error distribution
			normal_distribution<> dist(M_test_mean(0), M_test_sig(0));
			int sample = 0;
			for(; sample < samples; ++sample) {
				T sample_error = static_cast<T>(dist(gen));
				
				//Create a new stopping model and training with our previous iteration and our guess for the next iteration
				FStop<T> test_stop;
				test_stop.model.get_options() = model.get_options();
				test_stop.cost_model.get_options() = cost_model.get_options();
				
				test_stop.train(LaspMatrix<T>::hcat(iterations, LaspMatrix<T>(1,1,M_test_float)), LaspMatrix<T>::hcat(scores, LaspMatrix<T>(1,1,sample_error)), false);
				
				//Get the predicted second step from the guessed stopping model
				T M_double = test_stop.get_integer_step(upper_bound, error_thresh, error_prob);
				
				//Estimate the cost for the second step
				LaspMatrix<T> M_double_cost;
				cost_predict(LaspMatrix<T>(1,1,M_double), M_double_cost);
				
				//Add the difference to our estimate
				cost_val += M_test_cost(0) + M_double_cost(0) - M_best_cost;
				
				if (sample > 10 && (cost_val / sample) > std::min(best_cost_val, static_cast<T>(0.0))) {
					break;
				}
			}
			
			//Compute the averge cost difference
			cost_val /= sample;
			
			//cout << "Val: " << cost_val << " M_test: " << M_test << " mean: " << M_test_mean(0) << " sd: " << M_test_sig(0) << endl;
			
			//Set the best cost difference found
			if (cost_val < 0 && cost_val < best_cost_val) {
				best_cost_val = cost_val;
				best_test = M_test_float;
			}
		}
		
		//Return either M_best or the cheapest predicted interior step
		return best_cost_val == numeric_limits<T>::max() ? M_best_float : best_test;
	}
#endif
	
	template<class T>
T FStop<T>::stopping_value_gp(){

}

	template<class T>
T FStop<T>::stopping_value_heuristic(){
	if (scores.cols() < 2) {
		return 1;
	}
	
	return -(scores(scores.cols() - 1, 0) - scores(scores.cols() - 2, 0)) / (iterations(iterations.cols() - 1, 0) - (iterations(iterations.cols() - 2, 0)));
}
}
#endif
