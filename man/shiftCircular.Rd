% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{shiftCircular}
\alias{shiftCircular}
\title{Circular Shift for Neighbors Matrix}
\usage{
shiftCircular(X, Neighbors)
}
\arguments{
\item{\code{x}}{Numeric vector}

\item{\code{Neighbors}}{Integer vector with neighbors}
}
\value{
Matrix with Circular shifted versions of \code{x} defined by \code{Neighbors}
}
\description{
Shift a numeric vector by \code{shift}. End components 
is feedback into the begining. This will build a 
neighbors matrix
}

