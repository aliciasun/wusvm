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

#ifndef LASP_WUSVM_H
#define LASP_WUSVM_H

#ifdef __cplusplus
extern "C" {
#endif
	
	
	/*
		wusvm.h: External C/C++ interface for WU-SVM.
	 
		Documentation for all interface functions are provided below.
	 
		Example useage (in C):
	 
			 ...
			 wusvm_data train_data, test_data;
			 wusvm_model model;
			 wusvm_options options;
			 
			 create_options(&options);
			 set_option_int(options, 'v', 3);
			 set_option_double(options, 'g', 0.05);
			 set_option_double(options, 'c', 1.0);
			 set_option_int(options, 't', 1);
			 
			 create_data_from_file(&train_data, "a9a.txt");
			 fit_model(&model, train_data, options);
			 free_data(train_data);
			 
			 save_model(model, "a9a_test.model");
			 free_model(model);
			 load_model(&model, "a9a_test.model");
			 
			 create_data_from_file(&test_data, "a9a.t");
			 classify_data(test_data, model, options);
			 free_data(test_data);
			 free_model(model);
			 free_options(options);
			 ...
	*/
	
	
	

	typedef void* wusvm_model;
	typedef void* wusvm_data;
	typedef void* wusvm_options;
	
	//Type of kernel function to use in training and testing
	enum wusvm_kernel { RBF, LINEAR, POLYNOMIAL, SIGMOID };
	
	//Error codes returned by all of the following functions
	enum wusvm_error { NO_ERROR, TYPE_ERROR, ALLOCATION_ERROR, UNALLOCATED_POINTER_ERROR, ARGUMENT_ERROR, FILE_ERROR, NOT_FIT_ERROR };
	
	/* 
		create_data_from_array
	 
		Arguments:
			svm_data (output): Pointer to an unallocated wusvm_data object. After this function
								is executed, svm_data will point to a wusvm_data object.
	 
			x (input): Pointer to tightly packed data array (see description). Data in x is 
						copied, thus x may be safely deleted or reused.
	 
			y (input): Pointer to label array (see description). Data in y is also copied.
	 
			features (input): Number of feaures in each 'x' example.
	 
			points (input): Number of examples in both x and y.
	 
		Returns:
			Error code
	 
		Description:
			Create a data object using existing arrays. The 'x' array should be formatted as
			features by points in column major order. (The features of each training/testing
			example are contiguous in memory). The 'y' array should be a tighly packed
			array containing example labels as integers.
	*/
	int create_data_from_array(wusvm_data* svm_data, double* x, double* y, int features, int points);
	
	
	/*
		 create_test_data_from_array
		 
		 Arguments:
			 svm_data (output): Pointer to an unallocated wusvm_data object. After this function
			 is executed, svm_data will point to a wusvm_data object.
			 
			 x (input): Pointer to tightly packed data array (see description). Data in x is
			 copied, thus x may be safely deleted or reused.
				 
			 features (input): Number of feaures in each 'x' example.
			 
			 points (input): Number of examples in x.
		 
		 Returns:
			 Error code
		 
		 Description:
			 Has indentical behaivior to create_data_from_array, but creates data without specified
			 labels which may be classified by a previously fit model.
	 */
	int create_test_data_from_array(wusvm_data* svm_data, double* x, int features, int points);
	
	/*
		 create_data_from_file
		 
		 Arguments:
			 svm_data (output): Pointer to an unallocated wusvm_data object. After this function
			 is executed, svm_data will point to a wusvm_data object.
			 
			 file (input): Specifies file to load data from.
		 
		 Returns:
			Error code
		 
		 Description:
			 Creates a wusvm_data objects similarly to the two methods above, but reads the data
			 from the specified file which must be in the libsvm format.
	*/
	int create_data_from_file(wusvm_data* svm_data, const char* file);
	
	/*
		 get_labels_from_data
		 
		 Arguments:
			 svm_data (input): Previously created and classified wusvm_data object.
			 
			 labels (output): An (allocated!) array with a size of at least the number
								of points stored in svm_data.
		 
		 Returns:
			 Error code
		 
		 Description:
			 Retrives the output labels from a wusvm_data objects, whose examples have been
			 classified using classify_data (see below).
	 */
	int get_labels_from_data(wusvm_data svm_data, int* labels);
	
	/*
		 free_data
		 
		 Arguments:
			svm_data (input): Previously created wusvm_data object.
		
		 Returns:
			Error code
		 
		 Description:
			Safely de-allocates a previously created wusvm_data object.
	 */
	int free_data(wusvm_data svm_data);
	
	
	
	/*
		 create_options
		 
		 Arguments:
			 svm_options (output): Pointer to an unallocated wusvm_options object. After this function
			 is executed, svm_options will point to a wusvm_options object.
			
		 Returns:
			Error code
		 
		 Description:
			 Creates a options object with the default settings. Once created the options object can
			 be used to specify training and testing options to fit_model and classify_data. Options
			 can be changed using the set_* functions described below.
	 */
	int create_options(wusvm_options* svm_options);
	
	/*
		 set_kernel
		 
		 Arguments:
			 svm_options (input/output): A previously created options object to be modified.
	 
			 kernel (input): The kernel type to use in training/testing.
		 
		 Returns:
			Error code
		 
		 Description:
			 Sets the kernel type stored in the provided options object.
	 */
	int set_kernel(wusvm_options svm_options, wusvm_kernel kernel);
	
	/*
	 set_kernel
	 
	 Arguments:
		svm_options (input/output): A previously created options object to be modified.
	 
		args (input): The number of arguments to set.
	 
		options (input): An array of c-style strings specifying the arguments to set.
	 
	 Returns:
		Error code
	 
	 Description:
		Sets the options in the svm_options object as if args/options specified the 
		command line arguments passed into the train_mc/classify_mc program. Args and
		options correspond to the equivalent argc and argv. See the Readme for details
		on the effects of different arguments.
	*/
	int set_options(wusvm_options svm_options, int args, char** options);
	
	/*
		 set_option_int
		 
		 Arguments:
			svm_options (input/output): A previously created options object to be modified.
		 
			option (input): The integer or boolean option (argument) to set.
	 
			value (input): The value to assign to the given option. (Boolean options should
							use 1: true, 0: false).
		 
		 Returns:
			Error code
		 
		 Description:
			Sets an integer or boolean option to the specified value. See the Readme for details
			on the effects of different arguments. Arguments are identified by the option char.
	 */
	int set_option_int(wusvm_options svm_options, char option, int value);
	
	/*
		 set_option_double
		 
		 Arguments:
			 svm_options (input/output): A previously created options object to be modified.
			 
			 option (input): The integer or boolean option (argument) to set.
			 
			 value (input): The value to assign to the given option.
		 
		 Returns:
			 Error code
		 
		 Description:
			 Sets a floating point option to the specified value. See the Readme for details
			 on the effects of different arguments. Arguments are identified by the option char.
	 */
	int set_option_double(wusvm_options svm_options,char option, double value);
	
	/*
		 free_options
		 
		 Arguments:
			svm_options (input): Previously created wusvm_options object.
		 
		 Returns:
			Error code
		 
		 Description:
			Safely de-allocates a previously created svm_options object.
	 */
	int free_options(wusvm_options svm_options);
	
	/*
		 fit_model
		 
		 Arguments:
			model (output): Pointer to an unallocated wusvm_model object. After this function
							is executed, svm_data will point to a wusvm_model object.
	 
			svm_data (input): A fully loaded wusvm_data object to use as training data. (Must
								contain both examples and labels).
	 
			svm_options (input): A previously created wusvm_options object.
		 
		 Returns:
			Error code
		 
		 Description:
			Fits an SVM model using the provided data and options. Stores the resulting model in
			the given wusvm_model.
	*/
	int fit_model(wusvm_model* model, wusvm_data svm_data, wusvm_options svm_options);
	
	/*
		 save_model
		 
		 Arguments:
			 model (input): Previously created wusvm_options object.
	 
			 file (input): File to save the provided model to.
		 
		 Returns:
			 Error code
		 
		 Description:
			 Saves the given model to the specified file.
	*/
	int save_model(wusvm_model model, const char* file);
	
	/*
	 load_model
	 
	 Arguments:
		 model (output): Pointer to an unallocated wusvm_model object. After this function
						 is executed, svm_data will point to a wusvm_model object.
	 
		 file (input): File to load the model from.
	 
	 Returns:
		 Error code
	 
	 Description:
		 Loads in a model saved by save_model.
	 */
	int load_model(wusvm_model* model, const char* file);
	
	/*
		 free_model
		 
		 Arguments:
			model (input): Previously created wusvm_options object.
		 
		 Returns:
			Error code
		 
		 Description:
			Safely de-allocates a previously created wusvm_model object.
	 */
	int free_model(wusvm_model model);
	
	/*
		 classify_data
		 
		 Arguments:
			 test_data (input/output): A fully loaded wusvm_data object containing the test data to classify.
			 After the function, this object will also contain the classified labels.
	 
			 model (input): A previously fit wusvm_object used to classify data.
		 
			 svm_options (input): A previously created wusvm_options object.
		 
		 Returns:
			 Error code
		 
		 Description:
			 Classifies the data stored in test_data using the provided model and options. Classified labels are
			 stored back in test_data and may be retrieved using get_labels_from_data.
	 */
	int classify_data(wusvm_data test_data, wusvm_model model, wusvm_options svm_options);
	
#ifdef __cplusplus
}
#endif
#endif
