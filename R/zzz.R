.onLoad <- function(libname, pkgname) {
  packageStartupMessage("Welcome to Toom's Package")
  op <- options()
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("\n\nWelcome to Toom's Package\n")
}