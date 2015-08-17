// lasp_svm.cpp : Defines the entry point for the console application.
#include "lasp_svm.h"

int main(int argc, char* argv[])
{
  if (argc<3){
    return lasp::WRONG_NUMBER_ARGS;
  }
  lasp::svm_problem problem;
  lasp::parse_and_load(argc, argv, problem);

}

namespace lasp{

  int lasp_svm( svm_problem p ){
    float* x = sparse_to_full(p);
  
    return CORRECT;
  }

  int load_data(char *file, int& n, int& features, int*& y, svm_node**& x){
    cout << "loading data" << endl;
    ifstream fin;
    fin.open(file);
    features = 0;
    if (fin.is_open()){
      cout << file << " is open" << endl;
      n=0;
      char line[LINE_SIZE];

      while(fin.getline(line,LINE_SIZE)){// <----------------might be reading an extra line, could do better
        ++n;
      }
    
      y = new int[n];// times size of int?
      x = new svm_node*[n];// do we need to allocate a different amount of space for each array of nodes?
    
      fin.clear();
      fin.seekg(0 , ios::beg);

      int *ypos = y; //Probably could do this better
      svm_node** xpos= x;

      while(fin.getline(line,LINE_SIZE)){
        stringstream reader(line);

        reader >> *ypos;

        ypos++;//do we need to add the sizeof(int) instead?

        unsigned int features(0);
        string temp;//better way to do this, .good() was causing infinite loop
      
        while(reader>>temp){ //Definitely could be better
          ++features;
        }

        reader.clear();
        reader.seekg(2, ios::beg);
        svm_node* trainer= new svm_node[features + 1];//array of svm_nodes to put in x

        svm_node *tPos = trainer;
      
        while(reader>>temp){ //Definitely could be better
        
          svm_node node;
          char * temp2= new char[sizeof(temp)+1];
          copy(temp.begin(),temp.end(),temp2); //Could be better
          temp2[temp.size()]='\0';

          node.index=atoi(strtok (temp2, ":"));
          node.value=atof(strtok (NULL, ":"));

          features = node.index > features ? node.index : features;

          *tPos=node;
        
          tPos++;
          delete[] temp2;
        }
        svm_node end;
        end.index = -1;
        end.value = -1;
        *tPos = end;

        *xpos=trainer;
        xpos++;     
      }
      tempOutputCheck(x,y);
      return 0;
    }
    else{
      return UNOPENED_FILE_ERROR;
    }

  }

  int parse_and_load(int optc ,char ** opt, svm_problem& problem){
    int cur = FILE_IN;
    char* file;

    problem.options.nb_cand = 10;
    problem.options.set_size = -1;
    problem.options.maxiter = 20;
    problem.options.base_recomp = pow(2,0.25); 
    problem.options.verb = 1;
    problem.options.contigify = true;
    problem.options.maxnewbasis = 0;
    problem.options.stoppingcriterion = 5e-6;

    for(int i=1;i<optc;i++)
    {
      if(opt[i][0] != '-'){
        switch(cur){
        case FILE_IN:
          file = opt[i];
          break;
        case C_IN:
          problem.C = atoi(opt[i]);
          break;
        case GAMMA_IN:
          problem.gamma = atoi(opt[i]);
          break;
        default:
          fprintf(stderr,"Unknown input: %c\n", opt[i][1]);
          exit_with_help();
        }
        cur++;
        continue;
      }
      if(++i>=optc)
        exit_with_help();
      switch(opt[i-1][1])
      {
        case 'n': //sets nb_cand
          problem.options.nb_cand = atoi(opt[i]);
          break;
        case 's': // sets set_size
          problem.options.set_size = atoi(opt[i]);
          break;
        case 'i': // sets maxiter
          problem.options.maxiter = atoi(opt[i]);
          break;
        case 'b': //sets base_recomp
          problem.options.base_recomp = atof(opt[i]);
          break;
        case 'v': //sets verbosity
          problem.options.verb = atoi(opt[i]);
          break;
        case 'c': //sets contigify
          problem.options.contigify =  atoi(opt[i]);
          break;
        case 'm': //sets maxnewbasis
          problem.options.maxnewbasis = atoi(opt[i]);
          break;
        case 'x': //sets stoppingcriterion
          problem.options.stoppingcriterion = atof(opt[i]);
          break;
        default:
          fprintf(stderr,"Unknown option: -%c\n", opt[i-1][1]);
          exit_with_help();
      }
    }

    if(cur != DONE){
      exit_with_help();
    }

    int returnVal = load_data(file, problem.n, problem.features, problem.y, problem.x); //We should get the number of training examples from this
    if(returnVal != CORRECT){
      return returnVal;
    }

    if(problem.options.set_size < 0){
      problem.options.set_size = (int)ceil(problem.n/100.0);
    }

    tempOutputCheck(problem.x, problem.y);

    if(problem.options.verb > 0){
      cout << "nb_cand = " << problem.options.nb_cand << ", set_size = " << problem.options.set_size << ", maxiter = " << problem.options.maxiter << endl;
    }

    return CORRECT;
  }

  float* sparse_to_full(svm_problem p){
    float* x= new float[p.n * p.features];
    fill_n(x, p.n * p.features, 0); //Check if needed
    for (int i=0; i < p.n; ++i){
      for(int j= 0; p.x[i][j].index != -1; ++j){
        x[p.features * i + p.x[i][j].index] = p.x[i][j].value;
      }
    }
    return x;  
  }

  void exit_with_help(){}

  void tempOutputCheck(svm_node** x, int* y){
    ofstream out;
    out.open("outputTest.txt");
    for(int i = 0; i < 3; i++){
      out << "Node " << i / 3 << ": " << x[i / 3][i % 3].index << " " << x[i / 3][i % 3].value << ", Y: " << y[i/3] << endl;
    }
    out.close();
  }

}
