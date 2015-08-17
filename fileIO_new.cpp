#include "fileIO_new.h"


namespace lasp{
// the split() method is a
//shameles copy from https://www.safaribooksonline.com/library/view/c-cookbook/0596007612/ch04s07.html
void split(const string& s, char c, vector<string>& v) {
  string::size_type i = 0;
  string::size_type j = s.find(c);

  while (j != string::npos) {
    v.push_back(s.substr(i, j-i));
    i = ++j;
    j = s.find(c, j);

    if (j == string::npos)
      v.push_back(s.substr(i, s.length()));
  }
}

vector<string> split_on_delim(const string s, char delim){
  vector<string> subStrings;
  split(s,delim,subStrings);
  return subStrings;
}

  int load_LIBSVM(const char* filename, lasp::LaspMatrix<double>& X, lasp::LaspMatrix<double>& Y, int& n, int& d, bool dimUnknown){
  
  ifstream inputFile(filename);
  if (!inputFile.is_open()){
    cout << "ERROR: load_LIBSVM could not open the file: " << filename << endl;
  } 
  else{
    // if n and # of features is unknown determine them from the file
    if (dimUnknown){
      n = 0;
      d = 0;

      inputFile.clear();
      inputFile.seekg(0, ios::beg);

      string line;
    
      //read in each line
      while (getline(inputFile,line)){
        vector<string> subStrings = split_on_delim(line,' ');
        
        for (int i = 1; i < subStrings.size(); ++i){
          //cout << "i: " << i << " is " << subStrings[i] << endl;
          if (subStrings[i] != ""){
            vector<string> featurePair = split_on_delim(subStrings[i],':');
            int feature_index = atoi(featurePair[0].c_str());

            if (feature_index > d){
              d = feature_index;
            }

          }
        } 
      
      ++n;
       
      }
    }

    //Instantiate training data
    X = lasp::LaspMatrix<double>(n,d,0.0);
    //Instantiate training lables
    Y = lasp::LaspMatrix<double>(n,1,0.0);
    
    //ensure we are reading from the beginning of the file
    inputFile.clear();
    inputFile.seekg(0, ios::beg);

    int counter = 0;
    string line;
    
    //read in each line
    while (getline(inputFile,line) && counter < n){
      //cout << "===> FILEIO Counter: " << counter << endl;
      vector<string> subStrings = split_on_delim(line,' ');
      //cout << "I came" << endl;
      //sets the label of the training example
      double label = atof(subStrings[0].c_str());
      Y(counter,0) = label;
      //cout << "I labeled" << endl;
      
      //sets the value of each non-zero feature in the training example
      for (int i = 1; i < subStrings.size(); ++i){
	//cout << "i: " << i << " is " << subStrings[i] << endl;
	if (subStrings[i] != ""){
	  vector<string> featurePair = split_on_delim(subStrings[i],':');
	  int feature_index = atoi(featurePair[0].c_str());
	  double feature_value = atof(featurePair[1].c_str());
	  //cout << feature_index << "," << feature_value << endl;
	  X(counter,feature_index-1) = feature_value;	
	  //cout << X(counter,feature_index-1) << endl;
	}
      } 
      
      //cout << "I datad" << endl;
      ++counter;
       
    }
  }
  
  cout << "finished loading" << endl;
  
  return 0;

} 
  int load_LIBSVM(const char* filename, lasp::LaspMatrix<double>& X, lasp::LaspMatrix<int>& Y, int& n, int& d, bool dimUnknown){
  
  ifstream inputFile(filename);
  if (!inputFile.is_open()){
    cout << "ERROR: load_LIBSVM could not open the file: " << filename << endl;
  }
  else{
    // if n and # of features is unknown determine them from the file
    if (dimUnknown){
      n = 0;
      d = 0;

      inputFile.clear();
      inputFile.seekg(0, ios::beg);

      string line;
    
      //read in each line
      while (getline(inputFile,line)){
        vector<string> subStrings = split_on_delim(line,' ');
        
        for (int i = 1; i < subStrings.size(); ++i){
          //cout << "i: " << i << " is " << subStrings[i] << endl;
          if (subStrings[i] != ""){
            vector<string> featurePair = split_on_delim(subStrings[i],':');
            int feature_index = atoi(featurePair[0].c_str());

            if (feature_index > d){
              d = feature_index;
            }

          }
        } 
      
        ++n;
     
       }
    }
    //Instantiate training data
    X = lasp::LaspMatrix<double>(n,d,0.0);
    //Instantiate training lables
    Y = lasp::LaspMatrix<int>(n,1,0.0);
    
    //ensure we are reading from the beginning of the file
    inputFile.clear();
    inputFile.seekg(0, ios::beg);

    int counter = 0;
    string line;
    
    //read in each line
    while (getline(inputFile,line) && counter < n){
      //cout << "===> FILEIO Counter: " << counter << endl;
      vector<string> subStrings = split_on_delim(line,' ');
      //cout << "I came" << endl;
      //sets the label of the training example
      int label = atoi(subStrings[0].c_str());
      Y(counter,0) = label;
      //cout << "I labeled" << endl;
      
      //sets the value of each non-zero feature in the training example
      for (int i = 1; i < subStrings.size(); ++i){
	//cout << "i: " << i << " is " << subStrings[i] << endl;
	if (subStrings[i] != ""){
	  vector<string> featurePair = split_on_delim(subStrings[i],':');
	  int feature_index = atoi(featurePair[0].c_str());
	  double feature_value = atof(featurePair[1].c_str());
	  //cout << feature_index << "," << feature_value << endl;
	  X(counter,feature_index-1) = feature_value;	
	  //cout << X(counter,feature_index-1) << endl;
	}
      } 
      
      //cout << "I datad" << endl;
      ++counter;
       
    }
  }
  
  cout << "finished loading" << endl;
  
  return 0;

} 

}
