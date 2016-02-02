// if it is available, use OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif 
#include <thread>
#include <string> 
#include <RcppArmadillo.h>
#include <progress.hpp>
// [[Rcpp::depends(Rcpp,RcppArmadillo,RcppProgress)]]
// [[Rcpp::plugins(cpp11)]]

NumericVector SingleCircularShift(NumericVector x, int shift) {
  int N = x.size();
  int K = (shift > 0 ? (shift % N) : (shift==0 ? 0 : ((shift % N)+N)));
  if (K == 0) {
    return x;
  } else  { 
    NumericVector result(N);
    for (int i = 0; i < K;i++){
      result[i] = x[i+N-K];
    }
    for (int i = K; i < N;i++){
      result[i] = x[i-K];
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
NumericMatrix shiftCircular(NumericVector X, IntegerVector Neighbors) {
  int N = X.length();
  int M = Neighbors.length();
  NumericMatrix NeighborsMatrix(N,M);
  for(int i = 0; i < M; i++) {
      NeighborsMatrix(_,i) = SingleCircularShift(X, -Neighbors[i]);
  } 
  return NeighborsMatrix; 
}


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
    X = Xap*alphaVector + Xbp*betaVector + X*(1-alphaVector)*(1-betaVector);     
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
    Xbp = Xpi;
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
    Xbp = Xpi;
    Xap = pmax((1-Xi)*pmax((1-Xmi)*Xpi,(1-Xpi)*Xmi), (Xmi * Xpi));
    alphaVector = VectorSample(Size, AlphaProb);
    betaVector = VectorSample(Size, BetaProb);
    X = Xap*alphaVector + Xbp*betaVector + X*(1-alphaVector)*(1-betaVector);     
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

