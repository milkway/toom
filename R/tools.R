#' Parallel automata simulation
#' 
#' @param \code{Replications} Number of replications. Default 100.
#' @param \code{MaxIterations} Max number of iterations. Default 1000.
#' @param \code{Alpha} Alpha probability vector. Default seq(from = 0, to = 1, by = 0.25).
#' @param \code{Beta} Beta probability vector. Default seq(from = 0, to = 1, by = 0.25).
#' @param \code{Size} Size vector. Default c(1000).
#' @param \code{InitialProb} Initial probability vector. Default c(.5).
#' @return Dataframe. 
#' @export
simulation <- function(Replications = 100,
                       MaxIterations = 1000,
                       Alpha,
                       Beta,
                       Size = c(1000),
                       InitialProb = c(.5)
){
  stopifnot((Alpha <= 1), 
            (Alpha >= 0),
            (Beta <= 1),
            (Beta >= 0),
            (InitialProb <= 1), 
            (InitialProb >= 0))
  cl<-makeCluster(detectCores()-1)
  registerDoParallel(cl)
  Memory <- 
    foreach(p = InitialProb, .combine = 'rbind', .packages = c('foreach','toom')) %dopar% { 
      foreach(a = Alpha, .combine = 'rbind', .packages = c('foreach','toom')) %dopar% {
        foreach(b = Beta, .combine = 'rbind', .packages = c('foreach','toom')) %dopar% {
          foreach(j = 1:Replications, .combine = 'rbind', .packages = c('foreach','toom')) %dopar% {
            foreach(s = Size, .combine = 'rbind', .packages = c('foreach','toom')) %dopar% {
              doSimLast2(AlphaProb = a, BetaProb = b, Replication = j, Size = s, InitialProb = p, MaxIterations = MaxIterations, Neighbors = c(-1,0,1))
            }
          }
        }
      }
    }
  stopCluster(cl)
  return(Memory)  
}
