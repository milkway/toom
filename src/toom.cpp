// if it is available, use OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif 
// [[Rcpp::depends(Rcpp,RcppArmadillo,RcppProgress)]]
// [[Rcpp::plugins(cpp11,openmp)]]
#include <thread>
#include <string> 
#include <RcppArmadillo.h>
#include <progress.hpp>

class RNG
{
public:
  typedef std::mt19937 Engine;
  typedef std::uniform_real_distribution<double> Distribution;
  
  RNG() : engines(), distribution(0.0, 1.0)
  {
    int threads = std::max(1, omp_get_max_threads());
    for(int seed = 0; seed < threads; ++seed)
    {
      engines.push_back(Engine(seed));
    }
  }
  
  double operator()()
  {
    int id = omp_get_thread_num();
    return distribution(engines[id]);
  }
  
  std::vector<Engine> engines;
  Distribution distribution;
};






arma::ivec SingleCircularShift(arma::ivec x, int shift) {
  int N = x.n_elem;
  int K = (shift > 0 ? (shift % N) : (shift==0 ? 0 : ((shift % N)+N)));
  if (K == 0) {
    return x;
  } else  { 
    arma::ivec result(N);
    for (int i = 0; i < K;i++){
      result(i) = x(i+N-K);
    }
    for (int i = K; i < N;i++){
      result(i) = x(i-K);
    }
    return result;
  }
}

//' Circular Shift for Neighbors Matrix
//'
//' Shift a numeric vector by \code{shift}. End components 
//' is feedback into the begining. This will build a 
//' neighbors matrix
//' 
//' @param \code{x} Numeric vector
//' @param \code{Neighbors} Integer vector with neighbors
//' @return Matrix with Circular shifted versions of \code{x} defined by \code{Neighbors}
//' @export
// [[Rcpp::export]]
arma::imat shiftCircular(arma::ivec X, arma::ivec Neighbors) {
  int N = X.n_elem;
  int M = Neighbors.n_elem;
  arma::imat NeighborsMatrix(N,M);
  for(int i = 0; i < M; i++) {
    NeighborsMatrix.col(i) = SingleCircularShift(X, -Neighbors(i));
  } 
  return NeighborsMatrix; 
}

/*
//' Generate a random integer matrix
//' 
//' @param \code{n} Integer
//' @param \code{k} Integer
//' @param \code{p} Double in (0,1)
//' @return Random matrix
//' @export
// [[Rcpp::export]]
NumericMatrix MatrixSample(int n, int k, double p){
  try {
    if ((p < 0.0) | (p > 1)) {  // not the point
      throw std::range_error("Inadmissible value, pb in (0,1)");
    }
  } catch(std::exception &ex) {	
    forward_exception_to_r(ex);
  } catch(...) { 
    ::Rf_error("c++ exception (unknown reason)"); 
  }
  RNGScope tmp;
  NumericVector draws = floor(Rcpp::runif(n*k)+p);
  return NumericMatrix(n, k, draws.begin());
}
*/

/*
//' Generate a random integer vector
//' 
//' @param \code{n} Integer
//' @param \code{n} Double
//' @return Random vector
//' @export
// [[Rcpp::export]]
NumericVector VectorSample(int n, double p){
  try {
    if ((p < 0.0) | (p > 1)) {  // not the point
      throw std::range_error("Inadmissible value, pb in (0,1)");
    }
  } catch(std::exception &ex) {	
    forward_exception_to_r(ex);
  } catch(...) { 
    ::Rf_error("c++ exception (unknown reason)"); 
  }
  RNGScope tmp;
  NumericVector draws = floor(Rcpp::runif(n)+p);
  return draws;
}
*/

/*
//' Do a Simulation Step (Only 3 neighbors...)
//' 
//' @param \code{X} Numeric vector representing a configuration
//' @param \code{Alpha} Probability of change
//' @param \code{Neighbors} Integer vectors of neighbors
//' @return A List with new state and frequency of symbols
//' @export
// [[Rcpp::export]]
List simIteration(NumericVector X, double Alpha, IntegerVector Neighbors){
  int Size = X.length();
  NumericMatrix NeighborsMatrix = shiftCircular(X, Neighbors);
  NumericVector alphaVector = VectorSample(Size, Alpha);
  NumericVector Xmi = NeighborsMatrix( _, 0);
  NumericVector  Xi = NeighborsMatrix( _, 1);
  NumericVector Xpi = NeighborsMatrix( _, 2);
  NumericVector  Xp = pmax((1-Xi)*pmax(Xmi,Xpi), (Xmi * Xpi));
  NumericVector  Xr = Xp*alphaVector + X*(1-alphaVector);      
  NumericMatrix MXr = shiftCircular(Xr, Neighbors);
  double  f1 = mean(Xr);
  double  f0 = mean(1-Xr);
  double f00 = mean((1 - MXr(_,0)) * (1 - MXr(_,1)));
  double f01 = mean((1 - MXr(_,0)) * (    MXr(_,1)));
  double f10 = mean((    MXr(_,0)) * (1 - MXr(_,1)));
  double f11 = mean((    MXr(_,0)) * (    MXr(_,1)));
  return List::create(Named(   "X" ) = Xr,
                      Named(  "f0" ) = f0,
                      Named(  "f1" ) = f1,
                      Named( "f00" ) = f00,
                      Named( "f01" ) = f01,
                      Named( "f10" ) = f10,
                      Named( "f11" ) = f11); 
}
*/


/*
//' Do Automata Simulation (F fixed for Ramos & Leite)
//' 
//' @param \code{AlphaProb} Probability of change
//' @param \code{Replications} Number of replications. Default 100
//' @param \code{Size} Number of components in the configuration
//' @param \code{MaxIterations} Default 1000
//' @param \code{InitialProb} Probability for initial configuration
//' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
//' @return A data frame with Size, Replication, Iteration, AlphaProb and Frequency of Symbols
//' @export
// [[Rcpp::export]]
DataFrame doSim(double AlphaProb,
                int Replication = 100,
                int Size = 1000, 
                int MaxIterations = 1000, 
                double InitialProb = 0.5,
                IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
  NumericVector X = VectorSample(Size, InitialProb);
  NumericMatrix NeighborsMatrix = shiftCircular(X, Neighbors);
  NumericVector alphaVector = VectorSample(Size, AlphaProb);
  NumericVector Xmi = NeighborsMatrix( _, 0);
  NumericVector  Xi = NeighborsMatrix( _, 1);
  NumericVector Xpi = NeighborsMatrix( _, 2);
  NumericVector  Xp = pmax((1-Xi)*pmax(Xmi,Xpi), (Xmi * Xpi));
  NumericVector  f0(MaxIterations); 
  NumericVector  f1(MaxIterations); 
  NumericVector  f00(MaxIterations);
  NumericVector  f01(MaxIterations);
  NumericVector  f10(MaxIterations);
  NumericVector  f11(MaxIterations); 
  for (int i = 0; i < MaxIterations; i++){
    if (i % 1000 == 0) 
      Rcpp::checkUserInterrupt(); // Test if user wants stop proccess
    X = Xp*alphaVector + X*(1-alphaVector);     
    NeighborsMatrix = shiftCircular(X, Neighbors);
    Xmi = NeighborsMatrix( _, 0);
    Xi  = NeighborsMatrix( _, 1);
    Xpi = NeighborsMatrix( _, 2);
    f1[i]  = mean(X);
    f0[i]  = mean(1-X);
    f00[i] = mean((1 - Xmi) * (1 - X));
    f01[i] = mean((1 - Xmi) * (    X));
    f10[i] = mean((    Xmi) * (1 - X));
    f11[i] = mean((    Xmi) * (    X));
    Xp  = pmax((1-Xi)*pmax(Xmi,Xpi), (Xmi * Xpi));
    alphaVector = VectorSample(Size, AlphaProb);
  }
  IntegerVector Iterator = seq(1,MaxIterations); 
  NumericVector numIterator = as<NumericVector>(Iterator);
  // Cumulative Means
  NumericVector  f0cum = cumsum(f0);
  NumericVector  f1cum = cumsum(f1);
  NumericVector f00cum = cumsum(f00);
  NumericVector f01cum = cumsum(f01);
  NumericVector f10cum = cumsum(f10);
  NumericVector f11cum = cumsum(f11);
  return DataFrame::create(        _["Size"]  = Size,
                                  _["Alpha"]  = AlphaProb,
                            _["Replication"]  = Replication,
                               _["Iteration"] = numIterator,
                            _["SpaceMean_F0"] = f0,
                            _["SpaceMean_F1"] = f1,
                           _["SpaceMean_F00"] = f00,
                           _["SpaceMean_F01"] = f01,
                           _["SpaceMean_F10"] = f10,
                           _["SpaceMean_F11"] = f11,
                            _["TimeMean_F0"]  = f0cum/numIterator, 
                            _["TimeMean_F1"]  = f1cum/numIterator,
                           _["TimeMean_F00"]  = f00cum/numIterator,
                           _["TimeMean_F01"]  = f01cum/numIterator,
                           _["TimeMean_F10"]  = f10cum/numIterator,
                           _["TimeMean_F11"]  = f11cum/numIterator
  );
}
*/

/*
//' Do Automata Simulation (F fixed for Ramos & Leite 2015)
//' 
//' Save the space and temporal mean of configurations.
//' 
//' @param \code{AlphaProb} Probability of change
//' @param \code{Replications} Number of replications. Default 100
//' @param \code{Size} Number of components in the configuration
//' @param \code{MaxIterations} Default 1000
//' @param \code{InitialProb} Probability for initial configuration
//' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
//' @return A data frame with Size, Replication, Iteration, AlphaProb and Frequency of Symbols
//' @export
// [[Rcpp::export]]
DataFrame doSimLast(double AlphaProb,
                    int Replication,
                    int Size = 1000, 
                    int MaxIterations = 10000, 
                    double InitialProb = 0.5,
                    IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
  NumericVector X = VectorSample(Size, InitialProb);
  NumericMatrix NeighborsMatrix;
  NumericVector alphaVector;
  NumericVector Xmi;
  NumericVector  Xi;
  NumericVector Xpi;
  NumericVector  Xp;
  double  f0; 
  double  f0c = 0;   
  double  f1; 
  double  f1c = 0;   
  double  f00; 
  double  f00c = 0;   
  double  f01; 
  double  f01c = 0;
  double  f10; 
  double  f10c = 0;   
  double  f11; 
  double  f11c = 0;
  for (int i = 0; i < MaxIterations; i++){
    if (i % 1000 == 0) 
      Rcpp::checkUserInterrupt(); // Test if user wants stop proccess
    NeighborsMatrix = shiftCircular(X, Neighbors);
    Xmi = NeighborsMatrix( _, 0);
    Xi  = NeighborsMatrix( _, 1);
    Xpi = NeighborsMatrix( _, 2);
    f1  = mean(X);
    f0  = mean(1-X);
    f0c += f0;
    f1c += f1;
    f00 = mean((1 - Xmi) * (1 - X));
    f01 = mean((1 - Xmi) * (    X));
    f10 = mean((    Xmi) * (1 - X));
    f11 = mean((    Xmi) * (    X));
    f00c += f00;
    f01c += f01;
    f10c += f10;
    f11c += f11;
    // Next Grid Configuration
    Xp  = pmax((1-Xi)*pmax(Xmi,Xpi), (Xmi * Xpi));
    alphaVector = VectorSample(Size, AlphaProb);
    X = Xp*alphaVector + X*(1-alphaVector);     
  }
  return DataFrame::create(                _["Size"]  = Size,
                                          _["Alpha"]  = AlphaProb,
                                    _["Replication"]  = Replication,
                                       _["Iteration"] = MaxIterations,
                                    _["SpaceMean_F0"] = f0,
                                    _["SpaceMean_F1"] = f1,
                                   _["SpaceMean_F00"] = f00,
                                   _["SpaceMean_F01"] = f01,
                                   _["SpaceMean_F10"] = f10,
                                   _["SpaceMean_F11"] = f11,
                                    _["TimeMean_F0"]  = f0c/MaxIterations, 
                                    _["TimeMean_F1"]  = f1c/MaxIterations,
                                   _["TimeMean_F00"]  = f00c/MaxIterations,
                                   _["TimeMean_F01"]  = f01c/MaxIterations,
                                   _["TimeMean_F10"]  = f10c/MaxIterations,
                                   _["TimeMean_F11"]  = f11c/MaxIterations
  );
}
*/

/*
//' Do Automata Simulation (F with Alpha and Beta fixed for Ramos & Leite)
//' 
//' @param \code{AlphaProb} Probability of change 1
//' @param \code{BetaProb} Probability of change 2
//' @param \code{Replications} Number of replications. Default 100
//' @param \code{Size} Number of components in the configuration
//' @param \code{MaxIterations} Default 1000
//' @param \code{InitialProb} Probability for initial configuration
//' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
//' @return A data frame with Size, Replication, Iteration, AlphaProb and Frequency of Symbols
//' @export
// [[Rcpp::export]]
DataFrame doSim2(double AlphaProb,
                 double BetaProb,
                int Replication = 100,
                int Size = 1000, 
                int MaxIterations = 1000, 
                double InitialProb = 0.5,
                IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
  NumericVector X = VectorSample(Size, InitialProb);
  NumericMatrix NeighborsMatrix = shiftCircular(X, Neighbors);
  NumericVector alphaVector = VectorSample(Size, AlphaProb);
  NumericVector betaVector = VectorSample(Size, BetaProb);
  NumericVector Xmi = NeighborsMatrix( _, 0);
  NumericVector  Xi = NeighborsMatrix( _, 1);
  NumericVector Xpi = NeighborsMatrix( _, 2);
  NumericVector  Xbp = Xpi;
  NumericVector  Xap = pmax((1-Xi)*pmax((1-Xmi)*Xpi,(1-Xpi)*Xmi), (Xmi * Xpi));
  NumericVector  f0(MaxIterations); 
  NumericVector  f1(MaxIterations); 
  NumericVector  f00(MaxIterations);
  NumericVector  f01(MaxIterations);
  NumericVector  f10(MaxIterations);
  NumericVector  f11(MaxIterations); 
  for (int i = 0; i < MaxIterations; i++){
    if (i % 1000 == 0) 
      Rcpp::checkUserInterrupt(); // Test if user wants stop proccess
    //X = Xap*alphaVector + Xbp*betaVector + X*(1-alphaVector)*(1-betaVector);     
    //X = pmax(X, Xap*alphaVector);
    //X = pmax(X, Xbp*alphaVector);
    X = X*(1-alphaVector) +  Xap*alphaVector;
    X = X*(1-betaVector) + Xbp*betaVector;
    NeighborsMatrix = shiftCircular(X, Neighbors);
    Xmi = NeighborsMatrix( _, 0);
    Xi  = NeighborsMatrix( _, 1);
    Xpi = NeighborsMatrix( _, 2);
    f1[i]  = mean(X);
    f0[i]  = mean(1-X);
    f00[i] = mean((1 - Xmi) * (1 - X));
    f01[i] = mean((1 - Xmi) * (    X));
    f10[i] = mean((    Xmi) * (1 - X));
    f11[i] = mean((    Xmi) * (    X));
    Xbp = Xi;
    Xap = pmax((1-Xi)*pmax((1-Xmi)*Xpi,(1-Xpi)*Xmi), (Xmi * Xpi));
    alphaVector = VectorSample(Size, AlphaProb);
    betaVector = VectorSample(Size, BetaProb);
  }
  IntegerVector Iterator = seq(1,MaxIterations); 
  NumericVector numIterator = as<NumericVector>(Iterator);
  // Cumulative Means
  NumericVector  f0cum = cumsum(f0);
  NumericVector  f1cum = cumsum(f1);
  NumericVector f00cum = cumsum(f00);
  NumericVector f01cum = cumsum(f01);
  NumericVector f10cum = cumsum(f10);
  NumericVector f11cum = cumsum(f11);
  return DataFrame::create(        _["Size"]  = Size,
                                   _["Alpha"]  = AlphaProb,
                                   _["Beta"]  = BetaProb,
                                   _["Replication"]  = Replication,
                                   _["Iteration"] = numIterator,
                                   _["SpaceMean_F0"] = f0,
                                   _["SpaceMean_F1"] = f1,
                                   _["SpaceMean_F00"] = f00,
                                   _["SpaceMean_F01"] = f01,
                                   _["SpaceMean_F10"] = f10,
                                   _["SpaceMean_F11"] = f11,
                                   _["TimeMean_F0"]  = f0cum/numIterator, 
                                   _["TimeMean_F1"]  = f1cum/numIterator,
                                   _["TimeMean_F00"]  = f00cum/numIterator,
                                   _["TimeMean_F01"]  = f01cum/numIterator,
                                   _["TimeMean_F10"]  = f10cum/numIterator,
                                   _["TimeMean_F11"]  = f11cum/numIterator
  );
}
*/

/*
//' Do Automata Simulation (F with Alpha and Beta fixed for Ramos & Leite 2015)
//' 
//' Save the space and temporal mean of configurations.
//' 
//' @param \code{AlphaProb} Probability of change 1
//' @param \code{BetaProb} Probability of change 2
//' @param \code{Replications} Number of replications. Default 100
//' @param \code{Size} Number of components in the configuration
//' @param \code{MaxIterations} Default 1000
//' @param \code{InitialProb} Probability for initial configuration
//' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
//' @return A data frame with Size, Replication, Iteration, AlphaProb and Frequency of Symbols
//' @export
// [[Rcpp::export]]
DataFrame doSimLast2(double AlphaProb,
                     double BetaProb,
                    int Replication,
                    int Size = 1000, 
                    int MaxIterations = 10000, 
                    double InitialProb = 0.5,
                    IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
  NumericVector X = VectorSample(Size, InitialProb);
  //Rcout << "X: " << X << std::endl;
  NumericMatrix NeighborsMatrix;
  NumericVector alphaVector;
  NumericVector betaVector;
  NumericVector Xmi;
  NumericVector  Xi;
  NumericVector Xpi;
  NumericVector  Xap;
  NumericVector  Xbp;
  double  f0; 
  double  f0c = 0;   
  double  f1; 
  double  f1c = 0;   
  double  f00; 
  double  f00c = 0;   
  double  f01; 
  double  f01c = 0;
  double  f10; 
  double  f10c = 0;   
  double  f11; 
  double  f11c = 0;
  for (int i = 0; i < MaxIterations; i++){
    if (i % 1000 == 0) 
      Rcpp::checkUserInterrupt(); // Test if user wants stop proccess
    NeighborsMatrix = shiftCircular(X, Neighbors);
    Xmi = NeighborsMatrix( _, 0);
    Xi  = NeighborsMatrix( _, 1);
    Xpi = NeighborsMatrix( _, 2);
    f1  = mean(X);
    f0  = mean(1-X);
    f0c += f0;
    f1c += f1;
    f00 = mean((1 - Xmi) * (1 - X));
    f01 = mean((1 - Xmi) * (    X));
    f10 = mean((    Xmi) * (1 - X));
    f11 = mean((    Xmi) * (    X));
    f00c += f00;
    f01c += f01;
    f10c += f10;
    f11c += f11;
    // Next Grid Configuration
    Xbp = Xi;
    //Rcout << "Xbp: " << Xbp << std::endl;
    Xap = pmax((1-Xi)*pmax((1-Xmi)*Xpi,(1-Xpi)*Xmi), (Xmi * Xpi));
    //Rcout << "Xap: " << Xap << std::endl;
    alphaVector = VectorSample(Size, AlphaProb);
    //Rcout << "alphaVector: " << alphaVector << std::endl;
    betaVector = VectorSample(Size, BetaProb);
    //Rcout << "betaVector: " << betaVector << std::endl;
    //X = clamp(0,Xap*alphaVector + Xbp*betaVector + X*(1-alphaVector)*(1-betaVector),1);     
    //X = pmax(X, Xap*alphaVector);
    X = X*(1-alphaVector) +  Xap*alphaVector;
    //X = pmax(X, Xbp*alphaVector);
    X = X*(1-betaVector) + Xbp*betaVector;
    //Rcout << "X: " << X << std::endl;
  }
  return DataFrame::create(                _["Size"]  = Size,
                                           _["Alpha"]  = AlphaProb,
                                           _["Beta"]  =  BetaProb,
                                           _["Replication"]  = Replication,
                                           _["Iteration"] = MaxIterations,
                                           _["SpaceMean_F0"] = f0,
                                           _["SpaceMean_F1"] = f1,
                                           _["SpaceMean_F00"] = f00,
                                           _["SpaceMean_F01"] = f01,
                                           _["SpaceMean_F10"] = f10,
                                           _["SpaceMean_F11"] = f11,
                                           _["TimeMean_F0"]  = f0c/MaxIterations, 
                                           _["TimeMean_F1"]  = f1c/MaxIterations,
                                           _["TimeMean_F00"]  = f00c/MaxIterations,
                                           _["TimeMean_F01"]  = f01c/MaxIterations,
                                           _["TimeMean_F10"]  = f10c/MaxIterations,
                                           _["TimeMean_F11"]  = f11c/MaxIterations
  );
}
*/

/*
//' Do Automata Simulation (F with Alpha and Beta fixed for Ramos & Leite 2015)
//' 
//' Save the space and temporal mean of configurations.
//' 
//' @param \code{AlphaProb} Probability of change 1
//' @param \code{BetaProb} Probability of change 2
//' @param \code{Size} Number of components in the configuration
//' @param \code{MaxIterations} Default 1000
//' @param \code{InitialBlockSize} Inital block size. Default 1.
//' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
//' @return A value. If all zeros, return 0. If all ones, return 1. If MaxIterations reached, return -1.
//' @export
// [[Rcpp::export]]
int doSimGrid(double AlphaProb,
                     double BetaProb,
                     int Size = 1000, 
                     int MaxIterations = 100000, 
                     double InitialBlockSize = 1,
                     IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
  NumericVector X(Size);
  std::fill(X.begin(), X.end(), 0);
  std::fill_n(X.begin(), InitialBlockSize, 1);
  //Rcout << "Initial X: " << X << std::endl;
  NumericMatrix NeighborsMatrix;
  NumericVector alphaVector;
  NumericVector betaVector;
  NumericVector Xmi;
  NumericVector  Xi;
  NumericVector Xpi;
  NumericVector  Xap;
  NumericVector  Xbp;
  double  f0; 
  double  f1;
  int val = -1;
  for (int i = 0; i < MaxIterations; i++){
    //if (i % 1000 == 0) 
    //  Rcpp::checkUserInterrupt(); // Test if user wants stop proccess
    NeighborsMatrix = shiftCircular(X, Neighbors);
    Xmi = NeighborsMatrix( _, 0);
    Xi  = NeighborsMatrix( _, 1);
    Xpi = NeighborsMatrix( _, 2);
    //Rcout  << "NeighborsMatrix:\n " << NeighborsMatrix << std::endl;
    // Next Grid Configuration
    Xbp = Xi;
    //Rcout << "Xbp: " << Xbp << std::endl;
    Xap = pmax((1-Xi)*pmax((1-Xmi)*Xpi,(1-Xpi)*Xmi), (Xmi * Xpi));
    //Rcout << "Xap: " << Xap << std::endl;
    alphaVector = VectorSample(Size, AlphaProb);
    betaVector = VectorSample(Size, BetaProb);
    X = pmax(Xap*alphaVector, Xbp*betaVector);
    //Rcout << "X: " << X << std::endl;
    f1  = mean(X);
    //Rcout << "f1: " << f1 << std::endl;
    f0  = mean(1-X);
    //Rcout << "f0: " << f0 << std::endl;
    //Rcout << "i: " << i << std::endl;
    if (1 == f1) {
      val = 1;
      break;
    }
    if (1 == f0) {
      val = 0;
      break;
    }
  }
  return(val);
}
*/

/*
// [[Rcpp::export]]
DataFrame alpha_beta_grid(int threads=1,
                       int Step = 100,
                       int InitialBlockSize = 1, 
                       int MaxIterations = 100000, 
                       int Size = 1000,
                       IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
#ifdef _OPENMP
  if ( threads > 0 )
    omp_set_num_threads(threads);
    REprintf("Number of threads=%i\n", omp_get_max_threads());
#endif
  int size_grid = pow(Step,2);
 int long_dim = size_grid; //Replications*
  NumericMatrix result_matrix(long_dim, 3);
  result_matrix.fill(-2);
  //Progress p(size_grid, true);
    //Rcout << "\nReplication: " << r << std::endl;
    for (int i = 0; i < Step; i++) {
      for (int j = 0; j < Step; j++) {
        //p.increment();
        //Rcout << "j loop: " << j << std::endl;
        int I = i*Step + j;
        result_matrix(I, 0) = (double) i/Step;
        result_matrix(I, 1) = (double) j/Step;
        //result_matrix(i*Step + j, 2) = r;
        result_matrix(I, 2) =  doSimGrid((double) i/Step,(double) j/Step, Size, MaxIterations, InitialBlockSize, Neighbors);
      }
    }
    
  return DataFrame::create(      _["Alpha"]  = result_matrix(_,0),
                                  _["Beta"]  = result_matrix(_,1),
                                _["Status"]  = result_matrix(_,2));
}
 */
//alpha_beta_grid(threads = 1, Step = 20, InitialBlockSize = 1, MaxIterations = 10, Size = 10)
//alpha_beta_grid(threads = 1, Step = 4, Replications = 1, InitialBlockSize = 1, MaxIterations = 10, Size = 100)
//alpha_beta_grid(threads = 6, Step = 100, Replications = 200, InitialBlockSize = 1, MaxIterations = 1000000, Size = 1000)


/*
// [[Rcpp::export]]
arma::mat grid_simulation(int threads=1,
                          double step = 0.01,
                          int initial_block_lenght = 1, 
                          int max_iterations = 100000, 
                          int line_size = 1000,
                          IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
#ifdef _OPENMP
  if ( threads > 0 )
    omp_set_num_threads(threads);
  //srand(int(time(NULL)) ^ omp_get_thread_num());
  RNG rand;
  REprintf("Number of threads=%i\nn", omp_get_max_threads());
#endif
  int for_lenght = (int) 1/step;
  Rcout << "Lenght" << for_lenght << std::endl;
  int result_lenght = (int) pow(1/step,2);
  Rcout << "Result_lenght: " << result_lenght;
  arma::mat result_matrix(result_lenght, 3);
  result_matrix.fill(-2);
#pragma omp parallel for default(shared)
  for (int j=0; j<for_lenght; j++)
  {
#pragma omp simd
    for (int i=0; i<for_lenght; i++){
      arma::vec Automata =  {0,i*step,j*step,j*step,i*step,i*step,j*step,1};
      arma::ivec X(line_size);
      X.fill(0);
      std::fill_n(X.begin(), initial_block_lenght, 1);
      result_matrix(i*for_lenght+j, 0) = i*step;
      result_matrix(i*for_lenght+j, 1) = j*step;
      int count = 0;
      int val = -1;
      while ((val != 0)&&(val != 1)&&(count <= max_iterations)) {
        arma::imat NeighborsMatrix = shiftCircular(X, Neighbors);
        count++;
        for(int k = 0; k < line_size; k++){
          int Indice = NeighborsMatrix(k,0)*4 + NeighborsMatrix(k,1)*2 + NeighborsMatrix(k,2);
          //double r = (double) rand()/RAND_MAX;
          double r = rand();
          X(k) = (r <= Automata(Indice)) ? 1 : 0;
        }
        if (sum(X) == 0) val = 0;
        if (sum(X) == line_size) val = 1;
      }
      result_matrix(i*for_lenght+j, 2) = val;
    }
  }
  return result_matrix;
}
*/

/*
// [[Rcpp::export]]
arma::mat grid_simulation2(int threads=1,
                          double step = 0.01,
                          int initial_block_lenght = 1, 
                          int max_iterations = 100000, 
                          int line_size = 1000,
                          IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
#ifdef _OPENMP
  if ( threads > 0 )
    omp_set_num_threads(threads);
  //srand(int(time(NULL)) ^ omp_get_thread_num());
  RNG rand;
  REprintf("Number of threads=%i\nn", omp_get_max_threads());
#endif
  int for_lenght = (int) 1/step;
  Rcout << "Lenght" << for_lenght << std::endl;
  int result_lenght = (int) pow(1/step,2);
  Rcout << "Result_lenght: " << result_lenght;
  arma::mat result_matrix(result_lenght, 3);
  result_matrix.fill(-2);
#pragma omp parallel for default(shared) schedule(dynamic)
  for (int j=0; j<for_lenght; j++){
    for (int i=0; i<for_lenght; i++){
      arma::vec Automata =  {0,i*step,j*step,j*step,i*step,i*step,j*step,1};
      arma::ivec X(line_size);
      X.fill(0);
      std::fill_n(X.begin(), initial_block_lenght, 1);
      result_matrix(i*for_lenght+j, 0) = i*step;
      result_matrix(i*for_lenght+j, 1) = j*step;
      int count = 0;
      int val = -1;
      while ((val != 0)&&(val != 1)&&(count <= max_iterations)) {
        arma::imat NeighborsMatrix = shiftCircular(X, Neighbors);
        count++;
        for(int k = 0; k < line_size; k++){
          int Indice = NeighborsMatrix(k,0)*4 + NeighborsMatrix(k,1)*2 + NeighborsMatrix(k,2);
          //double r = (double) rand()/RAND_MAX;
          double r = rand();
          X(k) = (r <= Automata(Indice)) ? 1 : 0;
        }
        if (sum(X) == 0) val = 0;
        if (sum(X) == line_size) val = 1;
      }
      result_matrix(i*for_lenght+j, 2) = val;
    }
  }
  return result_matrix;
}
*/


 //' Do Automata Simulation (F with Alpha and Beta fixed for Ramos & Leite 2016)
 //' 
 //' Save the space and temporal mean of configurations.
 //' 
 //' @param \code{line_size} Number of components in the configuration
 //' @param \code{Step} Alpha and beta increment
 //' @param \code{max_iterations} Default 1000
 //' @param \code{replications} Default 1000
 //' @param \code{initial_block_size} Inital block size. Default 1.
 //' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
 //' @return Data frame. If all zeros, return 0. If all ones, return 1. If MaxIterations reached, return -1.
 //' @export
// [[Rcpp::export]]
Rcpp::DataFrame grid_simulation(int threads=1,
                           double step = 0.01,
                           int initial_block_lenght = 1, 
                           int max_iterations = 100000, 
                           int replications = 100,
                           int line_size = 1000,
                           IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
#ifdef _OPENMP
  if ( threads > 0 )
    omp_set_num_threads(threads);
  //srand(int(time(NULL)) ^ omp_get_thread_num());
  RNG rand;
  REprintf("Number of threads=%i\n\n", omp_get_max_threads());
#endif
  int for_lenght = (int) 1/step;
  //Rcout << "Lenght: " << for_lenght << std::endl;
  int result_lenght = (int) replications*pow(for_lenght,2);
  //Rcout << "Result_lenght: " << result_lenght << std::endl;
  arma::mat result_matrix(result_lenght, 2);
  arma::ivec result_status(result_lenght);
  arma::ivec result_replication(result_lenght);
  arma::ivec result_sum(result_lenght);
  arma::vec result_timesum(result_lenght);
  arma::ivec result_sum01(result_lenght);
  arma::vec result_timesum01(result_lenght);
  arma::ivec result_count(result_lenght);
  result_matrix.fill(0);
  result_replication.fill(0);
  result_status.fill(0);
  result_sum.fill(0);
  result_timesum.fill(0);
  result_sum01.fill(0);
  result_timesum01.fill(0);
  result_count.fill(0);
#pragma omp parallel for default(shared) schedule(dynamic)
  for(int r = 0; r < replications; r++){
    for(int j = 0; j<for_lenght; j++){
      for(int i = 0; i<for_lenght; i++){
        arma::vec Automata =  {0,i*step,j*step,j*step,i*step,i*step,j*step,1};
        arma::ivec X(line_size);
        X.fill(0);
        std::fill_n(X.begin(), initial_block_lenght, 1);
        result_matrix(r*pow(for_lenght,2) + i*for_lenght+j, 0) = i*step;
        result_matrix(r*pow(for_lenght,2) + i*for_lenght+j, 1) = j*step;
        result_replication(r*pow(for_lenght,2) + i*for_lenght+j) = r;
        int count = 0;
        int val = -1;
        int sumX = 0;
        double time_sumX = 0;
        double sumX01 = 0;
        double time_sumX01 = 0;
        while ((val != 0)&&(val != 1)&&(count <= max_iterations)) {
          arma::imat NeighborsMatrix = shiftCircular(X, Neighbors);
          count++;
          for(int k = 0; k < line_size; k++){
            int Indice = NeighborsMatrix(k,0)*4 + NeighborsMatrix(k,1)*2 + NeighborsMatrix(k,2);
            //double r = (double) rand()/RAND_MAX;
            double r = rand();
            X(k) = (r <= Automata(Indice)) ? 1 : 0;
          }
          sumX = sum(X);
          time_sumX += sumX;
          sumX01 = accu((1-SingleCircularShift(X, 1))%X);
          time_sumX01 += sumX01;
          if (sumX == 0) val = 0;
          if (sumX == line_size) val = 1;
        }
        result_status(r*pow(for_lenght,2) + i*for_lenght+j) = val;
        result_sum(r*pow(for_lenght,2) + i*for_lenght+j) = sumX;
        result_timesum(r*pow(for_lenght,2) + i*for_lenght+j) = time_sumX/(line_size*count);
        result_sum01(r*pow(for_lenght,2) + i*for_lenght+j) = sumX01;
        result_timesum01(r*pow(for_lenght,2) + i*for_lenght+j) = time_sumX01/(line_size*count);
        result_count(r*pow(for_lenght,2) + i*for_lenght+j) = count;
      }
    }    
  }
  Rcpp::NumericMatrix tmp = Rcpp::wrap(result_matrix);
  //return result_matrix;
  return DataFrame::create(      _["Alpha"]  = tmp(_,0),
                                 _["Beta"]  = tmp(_,1),
                                 _["Replication"]  = result_replication,
                                 _["Sum"]  = result_sum,
                                 _["TimeSum"]  = result_timesum,
                                 _["Sum01"]  = result_sum01,
                                 _["TimeSum01"]  = result_timesum01,
                                 _["Count"]  = result_count,
                                 _["Status"]  = result_status);
  
}
//Rcpp::checkUserInterrupt(); // Test if user wants stop proccess
//test <- grid_simulation3(threads = 4, step = .1, initial_block_lenght = 1, max_iterations = 1000, Replications = 1, line_size = 100)



// [[Rcpp::export]]
Rcpp::DataFrame grid_simulation2(int threads=1,
                                double step = 0.01,
                                int initial_block_lenght = 1, 
                                int max_iterations = 100000, 
                                int replications = 100,
                                int line_size = 1000,
                                IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
#ifdef _OPENMP
  if ( threads > 0 )
    omp_set_num_threads(threads);
  //srand(int(time(NULL)) ^ omp_get_thread_num());
  RNG rand;
  REprintf("Number of threads=%i\n\n", omp_get_max_threads());
#endif
  int for_lenght = (int) 1/step;
  //Rcout << "Lenght: " << for_lenght << std::endl;
  int result_lenght = (int) replications*pow(for_lenght,2);
  //Rcout << "Result_lenght: " << result_lenght << std::endl;
  arma::mat result_matrix(result_lenght, 2);
  arma::ivec result_status(result_lenght);
  arma::ivec result_replication(result_lenght);
  arma::ivec result_sum(result_lenght);
  result_matrix.fill(0);
  result_replication.fill(0);
  result_status.fill(0);
  result_sum.fill(0);
  for(int r = 0; r < replications; r++){
  #pragma omp parallel for default(shared) schedule(dynamic)    
    for(int j = 0; j<for_lenght; j++){
      for(int i = 0; i<for_lenght; i++){
        arma::vec Automata =  {0,i*step,j*step,j*step,i*step,i*step,j*step,1};
        arma::ivec X(line_size);
        X.fill(0);
        std::fill_n(X.begin(), initial_block_lenght, 1);
        result_matrix(r*pow(for_lenght,2) + i*for_lenght+j, 0) = i*step;
        result_matrix(r*pow(for_lenght,2) + i*for_lenght+j, 1) = j*step;
        result_replication(r*pow(for_lenght,2) + i*for_lenght+j) = r;
        int count = 0;
        int val = -1;
        int sumX = 0;
        while ((val != 0)&&(val != 1)&&(count <= max_iterations)) {
          arma::imat NeighborsMatrix = shiftCircular(X, Neighbors);
          count++;
          for(int k = 0; k < line_size; k++){
            int Indice = NeighborsMatrix(k,0)*4 + NeighborsMatrix(k,1)*2 + NeighborsMatrix(k,2);
            //double r = (double) rand()/RAND_MAX;
            double r = rand();
            X(k) = (r <= Automata(Indice)) ? 1 : 0;
          }
          sumX = sum(X);
          if (sumX == 0) val = 0;
          if (sumX == line_size) val = 1;
        }
        result_status(r*pow(for_lenght,2) + i*for_lenght+j) = val;
        result_sum(r*pow(for_lenght,2) + i*for_lenght+j) = sumX;
      }
    }    
  }
  Rcpp::NumericMatrix tmp = Rcpp::wrap(result_matrix);
  //return result_matrix;
  return DataFrame::create(      _["Alpha"]  = tmp(_,0),
                                 _["Beta"]  = tmp(_,1),
                                 _["Replication"]  = result_replication,
                                 _["Sum"]  = result_sum,
                                 _["Status"]  = result_status);
  
}


//' Do Automata Simulation (F with Alpha and Beta fixed for Ramos & Leite 2016)
//' 
//' Save the space and temporal mean of configurations.
//' 
//' @param \code{line_size} Number of components in the configuration
//' @param \code{Step} Alpha and beta increment
//' @param \code{max_iterations} Default 1000
//' @param \code{replications} Default 1000
//' @param \code{initial_block_size} Inital block size. Default 1.
//' @param \code{Neighbors} Integer vectors of neighbors. Defaulf c(-1,0,1)
//' @return Data frame. If all zeros, return 0. If all ones, return 1. If MaxIterations reached, return -1.
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame grid_simulation3(int threads=1,
                                double step = 0.01,
                                int initial_block_lenght = 1, 
                                int max_iterations = 100000, 
                                int line_size = 1000,
                                IntegerVector Neighbors = IntegerVector::create(-1,0,1)){
#ifdef _OPENMP
  if ( threads > 0 )
    omp_set_num_threads(threads);
  //srand(int(time(NULL)) ^ omp_get_thread_num());
  RNG rand;
  REprintf("Number of threads=%i\n\n", omp_get_max_threads());
#endif
  int for_lenght = (int) 1/step;
  //Rcout << "Lenght: " << for_lenght << std::endl;
  int result_lenght = (int) pow(for_lenght,2);
  //Rcout << "Result_lenght: " << result_lenght << std::endl;
  arma::mat result_matrix(result_lenght, 2);
  arma::ivec result_status(result_lenght);
  arma::ivec result_sum(result_lenght);
  arma::vec result_timesum(result_lenght);
  arma::ivec result_sum01(result_lenght);
  arma::vec result_timesum01(result_lenght);
  arma::ivec result_count(result_lenght);
  result_matrix.fill(0);
  result_status.fill(0);
  result_sum.fill(0);
  result_timesum.fill(0);
  result_sum01.fill(0);
  result_timesum01.fill(0);
  result_count.fill(0);
#pragma omp parallel for default(shared) schedule(dynamic)
    for(int j = 0; j<for_lenght; j++){
      for(int i = 0; i<for_lenght; i++){
        arma::vec Automata =  {0,i*step,j*step,j*step,i*step,i*step,j*step,1};
        arma::ivec X(line_size);
        X.fill(0);
        std::fill_n(X.begin(), initial_block_lenght, 1);
        result_matrix(i*for_lenght+j, 0) = i*step;
        result_matrix(i*for_lenght+j, 1) = j*step;
        int count = 0;
        int val = -1;
        int sumX = 0;
        double time_sumX = 0;
        double sumX01 = 0;
        double time_sumX01 = 0;
        while ((val != 0)&&(val != 1)&&(count <= max_iterations)) {
          arma::imat NeighborsMatrix = shiftCircular(X, Neighbors);
          count++;
          for(int k = 0; k < line_size; k++){
            int Indice = NeighborsMatrix(k,0)*4 + NeighborsMatrix(k,1)*2 + NeighborsMatrix(k,2);
            //double r = (double) rand()/RAND_MAX;
            double r = rand();
            X(k) = (r <= Automata(Indice)) ? 1 : 0;
          }
          sumX = sum(X);
          time_sumX += sumX;
          sumX01 = accu((1-SingleCircularShift(X, 1))%X);
          time_sumX01 += sumX01;
          if (sumX == 0) val = 0;
          if (sumX == line_size) val = 1;
        }
        result_status(i*for_lenght+j) = val;
        result_sum(i*for_lenght+j) = sumX;
        result_timesum(i*for_lenght+j) = time_sumX/(line_size*count);
        result_sum01(i*for_lenght+j) = sumX01;
        result_timesum01(i*for_lenght+j) = 2*time_sumX01/(line_size*count);
        result_count(i*for_lenght+j) = count;
      }
    }    
  Rcpp::NumericMatrix tmp = Rcpp::wrap(result_matrix);
  //return result_matrix;
  return DataFrame::create(      _["Alpha"]  = tmp(_,0),
                                 _["Beta"]  = tmp(_,1),
                                 _["Sum"]  = result_sum,
                                 _["TimeSum"]  = result_timesum,
                                 _["Sum01"]  = result_sum01,
                                 _["TimeSum01"]  = result_timesum01,
                                 _["Status"]  = result_status);
  
}

