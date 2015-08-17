//
//  exact_model.h
//  SP_SVM
//
//  Created by Nick Kolkin on 10/26/14.
//
//

#ifndef __APPROX_SVM__svm_model__
#define __APPROX_SVM__svm_model__

#include <iostream>
#include "base_model.h"
#include "optimize.h"
#include "lasp_func.h"


namespace lasp {

	//Gaussian process regression model
	template<class T>
class SVM_approx: public Model<T, int> {
protected:
				//Xin: training data Yin:labels K:store kernel matrix  
				 //IO:select the data points you want to train on
				//B:output  originalPosition
	LaspMatrix<T> Xin, Yin, K, I0, prevI0, B, originalPositions, smallB, beta, K2,basisIndex,Kji;
	int n, d, nOrig, numBasis,numAdditionalTr,itersize;
	T bias,lambda;
			double epsilon=0.0001; //stop criterion
			bool computeSmallB;

			opt& options(){
				return this->options_;
			}


		public:
			SVM_approx();
			SVM_approx(opt opt_in);

			LaspMatrix<T> get_hyp();
			vector<string> get_hyp_labels();
			int set_hyp(LaspMatrix<T> hyp);

			int init() {			
				numBasis=100;
				numAdditionalTr=30;
				itersize=10;
				computeSmallB=false;
				d = Xin.rows();
				n = Xin.cols();
				nOrig = n;
				originalPositions = LaspMatrix<T>(n,1,0.0);
				
				B = LaspMatrix<T>(numBasis,1,0.0);
				smallB=LaspMatrix<T>(numBasis,1,0.0,numBasis+numAdditionalTr,1);

				//for (int i = 0; i < n; i++) {
				//B(i,0) = 2*((double) std::rand() / (RAND_MAX)) - 1;
				//}
				//B.printMatrix("exactB");
				//#pragma omp parallel for
				prevI0 = LaspMatrix<T>(n,n,0.0);
				for (int i = 0; i < n; i++) {
					originalPositions(i) = i;
					prevI0(i,i) = 1.0;
				}
				I0.copy(prevI0);
				return 0;
			}

			virtual int train(LaspMatrix<T> X, LaspMatrix<int> y);//not defined yet
			int retrain(LaspMatrix<T> X, LaspMatrix<int> y);
			int train_internal();
			//added by Nick

			//everthing to do with evaluating the loss function, taking gradients, climbing candy mountain, etc.
			T evaluate();
			T evaluate(LaspMatrix<T> & gradient);
			T evaluate(LaspMatrix<T> & gradient, LaspMatrix<T> & hessian);
			T evaluate(LaspMatrix<T> & gradient, LaspMatrix<T> & hessian, bool computeGrad, bool computeHessian);
			T lossPrimal();
			LaspMatrix<T> gradientPrimal();
			LaspMatrix<T> hessianPrimal();

			//gardening section (pruning support vectors with weight zero for efficiency)
			int pruneData();

			//end

			virtual int predict(LaspMatrix<T> X, LaspMatrix<int>& output);
			int confidence(LaspMatrix<T> X, LaspMatrix<T>& output);
			int predict_confidence(LaspMatrix<T> X, LaspMatrix<int>& output_predictions, LaspMatrix<T>& output_confidence);
			int distance(LaspMatrix<T> X, LaspMatrix<T>& output);

			virtual T score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual);
			T loss(LaspMatrix<T> y_prob, LaspMatrix<int> y_actual);
			T loss(LaspMatrix<int> y_pred, LaspMatrix<T> y_prob, LaspMatrix<int> y_actual);

			T test(LaspMatrix<T> X, LaspMatrix<T> y);
			T test_loss(LaspMatrix<T> X, LaspMatrix<T> y);

			T operator()(LaspMatrix<T> w_in);
			T operator()(LaspMatrix<T> w_in, LaspMatrix<T>& grad);
			T operator()(LaspMatrix<T> w_in, LaspMatrix<T>& grad, LaspMatrix<T>& hess);

			vector<int> get_sv();
			void save(const char* output_file); 
		};

	//THESE ARE NOT TEMPLATED, I'M SORRY GABE :'(
	template <class T>
		T SVM_approx<T>::evaluate(){
			LaspMatrix<T> g,h;
			return evaluate(g, h, false, false);
		}

	template <class T>
		T SVM_approx<T>::evaluate(LaspMatrix<T> & gradient){
			LaspMatrix<T> h;
			return evaluate(gradient, h, true, false);
		}

	template <class T>
		T SVM_approx<T>::evaluate(LaspMatrix<T> & gradient, LaspMatrix<T> & hessian){
			return evaluate(gradient, hessian, true, true);
		}

	template <class T>
		T SVM_approx<T>::evaluate(LaspMatrix<T> & gradient, LaspMatrix<T> & hessian, bool computeGrad, bool computeHessian){
			//cout << "Pruning" << endl;
            //pruneData();
                        //this->B = this->B / (((LaspMatrix<T>)(this->B))(0,0));
			//cout << "COMPUTING LOSS" << endl;
			T loss = lossPrimal();
			//cout << "LOSS: " << loss << endl;
			//cout << "N: " << ((LaspMatrix<T>)sum(I0))(0,0) << endl;
			if (computeGrad){
			  //cout << "COMPUTING GRADIENT" << endl;
				gradient = gradientPrimal();
			} 
			if (computeHessian){
				cout << "COMPUTING HESSIAN" << endl;
				hessian = hessianPrimal();
			}
			
			//cout << "DONE EVALUATING" << endl;

			return loss;

		}



	template <class T>
		T SVM_approx<T>::lossPrimal(){
			cout<<"computing loss"<<endl;
			int n = this->n;
			int numBasis = this->numBasis;
			int numAdditionalTr = this->numAdditionalTr;
			T C = this->options().C;

			if(computeSmallB){
				//cout<<smallB.rows()<<" "<<smallB.cols()<<endl;
				LaspMatrix<T> KBt = this->K2*lasp::t(this->smallB);   //numBasis*1
				//KBt.printMatrix("Kbt");
				LaspMatrix<T> tempMatrix = (this->smallB*KBt);
				//tempMatrix.printMatrix("tempMatrix");
				if (tempMatrix.size() > 1) {
					cout << "WRONG SIZE" << endl;
					std::cin.ignore();
				}
				T term1 = tempMatrix(0,0);

				T term2 = 0;
				T temp,temp2;
				
				//prevI0.copy(I0);
				this->I0 = LaspMatrix<T>(numBasis+numAdditionalTr,numBasis+numAdditionalTr,0.0);
				//#pragma omp parallel for
				for (int i = 0; i < n; ++i){
					
					//temp=B*Kji*y
					temp=(this->smallB * this->Kji(i,0,i+1,n) * this->Yin(i,0))->eval().toNumber();
					temp2 = std::max(0.0,1.0-temp);
					if (temp >= 1.0 && temp2 != 0) {
						cout << "NOOOO" << endl;
						std::cin.ignore();
					}
					if (temp2 != 0){
						//this->I0(i,i) = 1.0;
						term2 += temp2 * temp2;
					//         //term2 += (1-temp) * (1-temp);
					// 	if (prevI0(i,i) == 0) {
					// 		//std::cin.ignore();
					// 	}
					}
					// if (this->prevI0(i,i) != this->I0(i,i)) {
					// 	//cout << "HOLY SHIT" << endl;
					// }
				 }
				if (term1 < 0) {
					cout << "BAD" << endl;
					cout<<term1<<endl;
					std::cin.ignore();
				}
				if (term2 < 0) {
					cout << "EVEN WORSE" << endl;
					std::cin.ignore();
				}
	            		//cout << "N: " << ((LaspMatrix<T>)sum(this->I0))(0,0) << endl;
				//cout<<term1<<" "<<term2<<endl;
				return 0.5*term1 + C*term2;

			}
			else{
				LaspMatrix<T> KBt = this->K2*lasp::t(this->B);
				LaspMatrix<T> tempMatrix = (this->B*KBt);
				cout<<"here"<<endl;
				if (tempMatrix.size() > 1) {
					cout << "WRONG SIZE" << endl;
					std::cin.ignore();
				}
				T term1 = tempMatrix(0,0);

				T term2 = 0;
				
				T temp,temp2;

				
				//prevI0.copy(I0);
				this->I0 = LaspMatrix<T>(numBasis,numBasis,0.0);
				//#pragma omp parallel for
				for (int i = 0; i < n; ++i){
			        //temp=B*Kji*y
			        //temp = KBt(0,i)*this->Yin(i,0);
					temp=(this->B * this->Kji(i,0,i+1,n) * this->Yin(i,0))->eval().toNumber();
					temp2 = std::max(0.0,1.0-temp);
					cout<<i<<endl;
					if (temp >= 1.0 && temp2 != 0) {
						cout << "NOOOO" << endl;
						std::cin.ignore();
					}
					if (temp2 != 0){
						//this->I0(i,i) = 1.0;
						term2 += temp2 * temp2;
					        //term2 += (1-temp) * (1-temp);
						// if (prevI0(i,i) == 0) {
						// 	//std::cin.ignore();
						// }
					}
					// if (this->prevI0(i,i) != this->I0(i,i)) {
					// 	//cout << "HOLY SHIT" << endl;
					// }
				}
				if (term1 < 0) {
					cout << "BAD" << endl;
					cout<<term1<<endl;
					std::cin.ignore();
				}
				if (term2 < 0) {
					cout << "EVEN WORSE" << endl;
					std::cin.ignore();
				}
	            		//cout << "N: " << ((LaspMatrix<T>)sum(this->I0))(0,0) << endl;
				//cout<<term1<<"  "<<term2<<endl;
				return 0.5*term1 + C*term2;
			}
		}

	template <class T>
		LaspMatrix<T> SVM_approx<T>::gradientPrimal(){
			cout<<"computing gradient"<<endl;
			T C = this->options().C;
		  //computeSmallB=added vectors 
			if(computeSmallB){
	      	//kji:numbasis*n    smallB:numbasis*1
	      	  LaspMatrix<T> KjiBt=t(this->Kji)*t(this->smallB);   //n*1
	      	  LaspMatrix<T> KBt = this->K2*t(this->smallB);

	      	  LaspMatrix<T> gradient = KBt + (2 * C * this->Kji * (KjiBt - t(this->Yin)));
	      	  return t(gradient);
	      	}
	      	else{
		      LaspMatrix<T> KBt = this->K2*t(this->B);  //numBasis*1
		      LaspMatrix<T> KjiBt=t(this->Kji)*t(this->B); //n*1
		      //we don't need I0 here?
		      LaspMatrix<T> gradient = KBt + (2 * C * this->Kji * (KjiBt - t(this->Yin))); //numBasis*1
		      return t(gradient);
		  }
		}

	template <class T>
		LaspMatrix<T> SVM_approx<T>::hessianPrimal(){
			T C = this->options().C;
			LaspMatrix<T> hessian = this->K2 + (2*C*this->K2*this->I0*this->K2);
            //this->B.printMatrix("exactB");
			return hessian;
		}


	//End not templated things

	template<class T>
		SVM_approx<T>::SVM_approx() {
			options().kernel = LINEAR;
		}

	template<class T>
		SVM_approx<T>::SVM_approx(opt opt_in) {
			options() = opt_in;
		}

	template <class T>
		LaspMatrix<T> SVM_approx<T>::get_hyp(){
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
		vector<string> SVM_approx<T>::get_hyp_labels(){
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
		int SVM_approx<T>::set_hyp(LaspMatrix<T> hyp){
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
		int SVM_approx<T>::train(LaspMatrix<T> X, LaspMatrix<int> y){

			LaspMatrix<T> temp = y.convert<T>();			

			//if (options().usebias && options().bias != 0) {
			//	Xin = LaspMatrix<T>::hcat(Xin, LaspMatrix<T>::vcat(X, LaspMatrix<T>::ones(X.cols(), 1)));
			//	bias = true;
			//} else {
			Xin = LaspMatrix<T>::hcat(Xin, X);
			//	bias = false;
			//}

			Yin = hcat(Yin, temp);
			cout << "INIT" << endl;
			this->init();
			cout << "TRAIN INTERNAL" << endl;
			return this->train_internal();
		}

	template <class T>
		int SVM_approx<T>::retrain(LaspMatrix<T> X, LaspMatrix<int> y){
			Xin = LaspMatrix<T>();
			Yin = LaspMatrix<T>();
			return train(X, y);
		}

	template<class T>
		int SVM_approx<T>::train_internal(){
			bool gpu = true;//options().usegpu;

			//Check that we have a CUDA device
			if (gpu && DeviceContext::instance()->getNumDevices() < 1) {
				if (options().verb > 0) {
					cerr << "No CUDA device found, reverting to CPU-only version" << endl;
				}

				options().usegpu = false;
				gpu = false;
			} else if(!gpu){
				DeviceContext::instance()->setNumDevices(0);
			} else if (options().maxGPUs > -1){
				DeviceContext::instance()->setNumDevices(options().maxGPUs);
			}

			//Timing Variables
			clock_t baseTime = clock();

			//kernel options struct for computing the kernel
			kernel_opt kernelOptions = options().kernel_options();

			//Training examples
			LaspMatrix<T> x = Xin;

			//Training labels
			LaspMatrix<T> y = Yin;
			cout << "TRANSFER TO GPU" << endl;
			//Move data to the gpu
			if (gpu) {
				int err = x.transferToDevice();
				err += y.transferToDevice();

				if (err != MATRIX_SUCCESS) {
					x.transferToHost();
					y.transferToHost();

					if (options().verb > 0) {
						cerr << "Device memory insufficient for data, reverting to CPU-only computation" << endl;
					}

					gpu = false;
				}
			}


			//basisIndex is a vector of length numBasis with each index randomly selected from a training matrix
			//additionally, we allocate memory equal to the number of additional vectors we will eventually use
			
			int numBasis = this->numBasis;
			int numAdditionalTr = this->numAdditionalTr;
			T initial_val=0;
			//basisIndex holds the initial basis vectors
			LaspMatrix<int> basisIndex=LaspMatrix<int>(1,numBasis,initial_val,1,numBasis+numAdditionalTr);
			// //a indicator to remember which points have been selected as basis vector
			// LaspMatrix<T> basisIO=LaspMatrix<T>(1,this->n,initial_val,1,this->n);

			LaspMatrix<int> validIndices = LaspMatrix<int>(this->n,1);
			int numValidIndices = this->n;

			for(int i=0;i<n;i++){
				validIndices(i)=i;
			}
			validIndices.shuffle();

			for(int i=0;i<numBasis;i++){
				basisIndex(i)=validIndices(i);
			}

			//this is actually grabbing all the data for initial basis vectors from the training data
			LaspMatrix<T> subSection = LaspMatrix<T>(numBasis,this->d);
			for(int i=0;i<numBasis;i++){
				subSection.setCol(i,x,basisIndex(i));
			}


			
			//this is computing the kernel for everything at the beginning
			//cout << "COMPUTING KERNEL" << endl;
			// this->K = LaspMatrix<T>(n,n,0.0);
			// #pragma omp parallel for
			// for (int i = 0; i < this->n; i++) {
			// 	for (int j = i; j < this->n; j++) {
			// 		T sum = 0;
			// 		for (int k = 0; k < this->d; k++) {
			// 			sum += x(i,k) * x(j,k);
			// 		}
			// 		this->K(j,i) = sum;
			// 		this->K(i,j) = sum;
			// 	}
			// }
			


			int currentNumBasis=numBasis;
			//basisIndex.printMatrix("initial basis vector index");

			//prep the sub kernels before the loop
			//K2 is the basis vectors kernel with additional memory space for additional vector
			this->K2 = LaspMatrix<T>(numBasis,numBasis,0.0,numBasis+numAdditionalTr,numBasis+numAdditionalTr);
			//Kji holds the kernel between each training examples and all basis vectors, this could be used when compute loss
			this->Kji=LaspMatrix<T>(n,numBasis,0.0,n,numBasis+numAdditionalTr);
			std::cin.ignore();


			LaspMatrix<T> tempK;
			//K.gather(Kji,basisIndex);
			//Kji=t(Kji);
			//Kji.gather(K1,basisIndex);
			//Kji.gather(K2,basisIndex);




			
			// this commented out code only computes the kernel for the initial basis vectors
			// this will be implemented as an option to compute the kernel as you go to save memory
			cout << "COMPUTING KERNEL" << endl;
			//this->K = LaspMatrix<T>(numBasis,numBasis,0.0);
			#pragma omp parallel for
			for (int i = 0; i < this->numBasis; i++) {
				for (int j = 0; j < this->n; j++) {
					T sum = 0;
					for (int k = 0; k < this->d; k++) {
						sum += subSection(i,k) * x(j,k);
					}
					this->Kji(j,i) = sum;
					//this->K2(j,i) = sum;
				}
			}
			//Kji.printMatrix("Kji");
			Kji.gather(K2,basisIndex);
			//K2.printMatrix("K2");
			cout << "KERNEL COMPUTED" << endl;

			//Move support vector stuff to gpu
			if (gpu) {
				B.transferToDevice();
				K2.transferToDevice();
				smallB.transferToDevice();
			}
			
			//training SVM on initial basis vectors
			//cout << "CONJUGATE OPTIMIZING!!!" << endl;
			optimize_options opt_opt = optimize_options();
			opt_opt.maxIter = 5000;
			opt_opt.epsilon = .0001;
			opt_opt.gpu = gpu;
			computeSmallB=false;
			cout<<"before optimize"<<endl;
			conjugate_optimize(*this, B, opt_opt);
			cout<<"after optimize"<<endl;
			B.printMatrix("B");
			T trainingError = lossPrimal();
			T mostRecentTrainingError = trainingError*2;
			

			computeSmallB=true;
			//heuristically add more basis vectors
			int maxIterations = 5;
			int iterationsCount = 0;
			cout<<"before iterative training"<<endl;

			
			for(int iter=0;iter<numAdditionalTr;iter++){
				while(std::abs(trainingError-mostRecentTrainingError)>epsilon && iterationsCount<maxIterations){
					iterationsCount++;
					cout<<"we on iteration "<<iter<<endl;
					mostRecentTrainingError=trainingError;
	                //randomly choose next itersize columns
					//select new basis vector as a group of itersize
					LaspMatrix<int> tempBasis=LaspMatrix<int>(1,itersize,0);

	                //validIndices.printMatrix("validIndices");
					tempBasis=validIndices.getSubMatrix(iter*itersize+numBasis,0,numBasis+(iter+1)*itersize,1);
					LaspMatrix<int> tempIndices = LaspMatrix<int>(numBasis+iter+1,1,0.0);
					for(int i=0;i<numBasis+iter;i++){
						tempIndices(i)=basisIndex(i);
					}


					
					T minVal = 1000000;
					int minIndex = -1;
					for(int i=0;i<itersize;i++){


						tempIndices(numBasis+iter)=tempBasis(i);
						tempIndices.printMatrix("tempIndices");
						LaspMatrix<T> temp = LaspMatrix<T>(numBasis+iter+1,1,0.0);
						//cout<<temp.rows()<<" "<<temp.cols()<<endl;
						LaspMatrix<T> temp2;
						Kji.resize(n,numBasis+iter+1);
						//compute kernel for the new basis vector
						for (int j = 0; j < this->n; j++) {
							T sum = 0;
							for (int k = 0; k < this->d; k++) {
								sum += x(tempBasis(i),k) * x(j,k);
							}
							this->Kji(j,numBasis+iter) = sum;
						}
						//K.getSubMatrix(0,tempIndices(numBasis+iter),K.cols(),tempIndices(numBasis+iter)+1).gather(temp,tempIndices);
					    Kji.printMatrix("new Kji");

						K2.resize(numBasis+iter+1,numBasis+iter+1);
						Kji.gather(K2,tempIndices);
						//K2.setCol(numBasis+iter,temp);

						//LaspMatrix<T> tempt;
						//temp.transpose(tempt);
						//K2.setRow(numBasis+iter,tempt);
						
		            	//cout << "CONJUGATE OPTIMIZING!!!" << endl;
						optimize_options opt_opt = optimize_options();
						opt_opt.maxIter = 5000;
						opt_opt.epsilon = .0001;
						//opt_opt.gpu = gpu;
						smallB=LaspMatrix<T>(numBasis+iter+1,1,0.0);
						
						//temp2=K.getSubMatrix(0,tempIndices(numBasis+iter),K.cols(),tempIndices(numBasis+iter)+1);
						//Kji.setRow(numBasis+iter,temp2);
                         K2.printMatrix("new K2");
                         

						smallB.printMatrix("smallB before");
                        //Kji.printMatrix("Kji");

					   //learn weight for each individually

						conjugate_optimize(*this, smallB, opt_opt);
					   // smallB.printMatrix("smallB after");
					  // cout<<"after"<<smallB.rows()<<" "<<smallB.cols()<<endl;
						T tempLoss = lossPrimal();
						cout<<"temploss "<< tempLoss<<endl;
						std::cin.ignore();
						if(tempLoss<minVal){
							minVal = tempLoss;
							minIndex = tempBasis(i);

						}
	                  //cout<<"we on inner iteration"<<i<<endl;

					}
					cout<<minIndex<<endl;
	                //this is a temp way around, how do we use the memory preallocated to grow the matrix?
					LaspMatrix<int> oneIndex=LaspMatrix<int>(1,1,minIndex);
	                //basisIndex.resize(1,numBasis+iter+);
					basisIndex=vcat(basisIndex,oneIndex);
					trainingError=minVal;
	                //oneIndex.printMatrix("One index");
	               // basisIndex.printMatrix("basis after select");



				}
			}
	            //redo last change
			if(iterationsCount==1){
				beta=B;
			}
			else{
				smallB.resize(numBasis+iterationsCount-1,1);
				beta=smallB;
			}


                //what is this?????
			double diff;
			LaspMatrix<T> temp;
			LaspMatrix<T> grad;
			grad = gradientPrimal();
			grad = t(grad);
			grad.printMatrix("grad");
			grad.colSqSum(temp);
			diff = temp(0,0);
			if (temp.size() > 1) {
				cout << "BAD" << endl;
			}
			cout << "diff: " << diff << endl;


			

			if (gpu) {
				beta.transferToHost();
			}
			beta.printMatrix("approxB");
			if (options().verb > 0){
				cout << "Training Complete" << endl;
			}
			cout << "DONE" << endl;

			return 0;
		}

	template <class T>
		int SVM_approx<T>::predict(LaspMatrix<T> X, LaspMatrix<int>& output){
			output = LaspMatrix<int>(Xin.cols(),1,0);
           // #pragma omp parallel for
			for (int i = 0; i < X.cols(); i++) {
				T score = 0;
				for (int j = 0; j < B.size(); j++) {
					LaspMatrix<T> temp = ( t(csel(Xin,j)) * csel(X,i) );
					score += B(j) * temp(0,0);
				}
				output(i,0) = ((0 < score) - (score < 0));
			}
			return 0;
		}

	template <class T>
		int SVM_approx<T>::confidence(LaspMatrix<T> X, LaspMatrix<T>& output){
			return -1;
		}

	template <class T>
		int SVM_approx<T>::predict_confidence(LaspMatrix<T> X, LaspMatrix<int>& output_predictions , LaspMatrix<T>& output_confidence){
			return -1;
		}

	template <class T>
		int SVM_approx<T>::distance(LaspMatrix<T> X, LaspMatrix<T>& output){
			return -1;
		}

	template <class T>
		T SVM_approx<T>::score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual){
			double sum = 0;
//#pragma omp paralell for
			for (int i = 0; i < y_pred.size(); i++) {
				if (y_pred(i,0) == y_actual(i,0)) {
					sum += 1.0;
				}
			}
			return sum / y_pred.size();
		}

	template <class T>
		T SVM_approx<T>::loss(LaspMatrix<T> y_prob, LaspMatrix<int> y_actual){
			return -1;
		}

	template <class T>
		T SVM_approx<T>::loss(LaspMatrix<int> y_pred, LaspMatrix<T> y_prob, LaspMatrix<int> y_actual){
			return -1;
		}

	template <class T>
		T SVM_approx<T>::test(LaspMatrix<T> X, LaspMatrix<T> y){
			return -1;
		}

	template <class T>
		T SVM_approx<T>::test_loss(LaspMatrix<T> X, LaspMatrix<T> y){
			return -1;
		}

	template<class T>
		vector<int> SVM_approx<T>::get_sv() {
			return originalPositions;
		}


	template<class T>
		T SVM_approx<T>::operator()(LaspMatrix<T> b_in){
			if(computeSmallB){
				this->smallB=b_in;
			}
			else{
				this->B = b_in;
			}
			return evaluate();
		}

	template<class T>
		T SVM_approx<T>::operator()(LaspMatrix<T> b_in, LaspMatrix<T>& grad){
			if(computeSmallB){
				this->smallB = b_in;
			}
			else{
				this->B = b_in;
			}
			return evaluate(grad);

		}

	template<class T>
		T SVM_approx<T>::operator()(LaspMatrix<T> b_in, LaspMatrix<T>& grad, LaspMatrix<T>& hess){
			if(computeSmallB){
				this->smallB = b_in;
			}
			else{
				this->B=b_in;
			}
			return evaluate(grad, hess);
		}

	template<class T>
		void SVM_approx<T>::save(const char* output_file){

		}

	}


#endif /* defined(__SP_SVM__svm_model__) */
