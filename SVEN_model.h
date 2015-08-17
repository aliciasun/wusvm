//
//  SVEN_model.h
//  SP_SVM
//
//  Created by Nick Kolkin on 10/26/14.
//
//

#ifndef __SVEN_SVM__svm_model__
#define __SVEN_SVM__svm_model__

#include "exact_model.h"
#include "lasp_func.h"

namespace lasp {
    
    //Gaussian process regression model
    template<class T>
    class SVEN_model: public SVM_exact<T> {
    protected:
        double t;
        int dOld;

        LaspMatrix<T> YOld;
    public:
        SVEN_model(double t, double lambda)
        : SVM_exact<T>() {
            this->t = t;
	    this->lambda = lambda;
            this->options().C = 1/(2*lambda);
        }
        SVEN_model(opt opt_in, double t, double lambda)
        : SVM_exact<T>(opt_in) {
            this->t = t;
	    this->lambda = lambda;
            this->options().C = 1/(2*lambda);
        }
        int train(LaspMatrix<T> X, LaspMatrix<T> y);
        virtual int train(LaspMatrix<T> X, LaspMatrix<int> y);
        virtual int predict(LaspMatrix<T> X, LaspMatrix<int>& output);
        virtual T score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual);
        int transformTo();
        int transformBack();
        int write_to_csv(string,LaspMatrix<T>);
    };


    
    template <class T>
    int SVEN_model<T>::train(LaspMatrix<T> X, LaspMatrix<int> y){
	return 0;
    }

    template <class T>
    int SVEN_model<T>::train(LaspMatrix<T> X, LaspMatrix<T> y){
        
        
        //LaspMatrix<T> temp = y.convert<T>();
        
        //if (this->options().usebias && this->options().bias != 0) {
        //    Xin = LaspMatrix<T>::hcat(Xin, LaspMatrix<T>::vcat(X, LaspMatrix<T>::ones(X.cols(), 1)));
        //    this->bias = true;
        //} else {
        this->Xin = LaspMatrix<T>::hcat(this->Xin, X);
        //    this->bias = false;
        //}
        
        this->Yin = hcat(this->Yin, y);
	//write_to_csv("data2.csv",this->Xin);
	cout << this->options().C << endl;
        transformTo();
        this->init();
        this->train_internal();
        transformBack();
        this->B.printMatrix("SVEN_B");
        write_to_csv("betas.txt",this->B);
        return 0;
    }
    
    template <class T>
    int SVEN_model<T>::transformTo() {
        this->n = this->Yin.size();
        this->d = this->Xin.rows();
        int n = this->n;
        int d = this->d;
        LaspMatrix<T> Xin = this->Xin;
        LaspMatrix<T> Yin = this->Yin;
        
        //matrix of ones for copying y1 for each feature
        lasp::LaspMatrix<T> ones_y1 = LaspMatrix<T>(1,this->d,1.0);
        
        //y1 copied for each feature
        lasp::LaspMatrix<T> y1_big = ones_y1 * (Yin / this->t);
	//lasp::LaspMatrix<T> temp = (Yin / this->t);
	//write_to_csv("y_t.csv",temp);
        
        lasp::LaspMatrix<double> x_plus = Xin+y1_big;
        lasp::LaspMatrix<double> x_minus = Xin-y1_big;
        //Transformed Input Data
        this->Xin = lasp::t(vcat(x_minus, x_plus));
        
        //transformed_data = t(transformed_data);
        this->Yin = hcat(LaspMatrix<T>(d,1,1.0), LaspMatrix<T>(d,1,-1.0));
        
        this->n = 2*d;
        this->d = n;
        dOld = d;
        this->YOld.copy(this->Yin);
        //this->Xin.printMatrix("XIN");
        //write_to_csv("data.csv", this->Xin);
        //write_to_csv("labels.csv", this->Yin);
        //write_to_csv("labels2.csv", this->YOld);
        return 0;
    }
    
    template <class T>
    int SVEN_model<T>::transformBack() {
	//this->B.printMatrix("B");
        int d = this->d;
	//cout << d << endl;
        T C = this->options().C;
        LaspMatrix<T> temp = LaspMatrix<T>(2*dOld,1,0.0);
        LaspMatrix<T> w = LaspMatrix<T>(1,d,0.0);
	//cout << this->Xin.rows() << endl;
	//cout << this->Xin.cols() << endl;
	//cout << this->B.rows() << endl;
	//cout << this->B.cols() << endl;
        //LaspmMatrix<T> temp2 = LaspMatrix<T>(this->B.size,d,0.0);
//#pragma omp parallel for
        for (int i = 0; i < this->B.size(); i++) {
	  //cout << i << endl;
            w = w + this->B(i,0) * csel(this->Xin,i); //wrong indices when pruning
        }
  

	//cout << d << endl;
        //w.printMatrix("W");
        //write_to_csv("out.csv", w);
        //this->Xin.printMatrix("XIN");
        //YOld.printMatrix("YOLD");
        LaspMatrix<T> xw = lasp::t(this->Xin) * w;
        //xw.printMatrix("xw");
	//cout << d << endl;
        temp = 1 - mul(lasp::t(YOld), (lasp::t(this->Xin) * w));
        temp = lasp::t(temp);
        //temp.printMatrix("TEMP");
        for (int i = 0; i < 2*dOld; i++) {
            if (temp(i,0) < 0) {
                temp(i,0) = 0.0;
            } else {
                temp(i,0) = C * temp(i,0);
            }
        }
        //temp.printMatrix("TEMP");
        
        T s = ((LaspMatrix<T>)(sum(temp)))(0,0);
        T ksi = this->t / s;
        lasp::LaspMatrix<T> betas = LaspMatrix<T>(dOld,1,0.0);
        
	//cout << d << endl;
        #pragma omp parallel for
        for (int i = 0; i < 2*dOld; ++i) {
            T pos = this->originalPositions(i);
            if (pos < dOld) {
                betas(pos) += temp(i) * ksi;
            } else {
                betas(pos - dOld) -= temp(i) * ksi;
            }
        }
        this->B = betas;
        return 0;
    }
    
    template <class T>
    int SVEN_model<T>::predict(LaspMatrix<T> X, LaspMatrix<int>& output){
        LaspMatrix<T> preds = this->B * X;
        output = LaspMatrix<int>(X.cols(),1,0.0);
        #pragma omp parallel for
        for (int i = 0; i < X.cols(); i++) {
            T score = preds(i,0);
            output(i,0) = ((0 < score) - (score < 0));
        }
        return 0;
    }
    
    template <class T>
    T SVEN_model<T>::score(LaspMatrix<int> y_pred, LaspMatrix<int> y_actual){
        double sum = 0;
        #pragma omp paralell for
        for (int i = 0; i < y_pred.size(); i++) {
            if (y_pred(i,0) == y_actual(i,0)) {
                sum += 1.0;
            }
        }
        return sum / y_pred.size();
    }
    
    template <class T>
    int SVEN_model<T>::write_to_csv(string filename,LaspMatrix<T> data) {
        ofstream f(filename);
        for (int i = 0; i < data.cols(); i++) {
            for (int j = 0; j < data.rows(); j++) {
                f << data(i,j) << ",";
            }
            f << "\n";
        }
        f.close();
        return 0;
    }
}

#endif
