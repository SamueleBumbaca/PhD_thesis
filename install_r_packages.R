# Install required packages for VS Code R extension
cat("Installing languageserver from CRAN...\n")
install.packages('languageserver', repos='https://cran.rstudio.com/', dependencies=TRUE)

cat("Installing remotes package (needed for GitHub installation)...\n")
install.packages('remotes', repos='https://cran.rstudio.com/', dependencies=TRUE)

cat("Installing httpgd from GitHub (not available on CRAN for R 4.5.1)...\n")
remotes::install_github("nx10/httpgd")

# Check if packages are installed
if(require("languageserver", quietly = TRUE) && require("httpgd", quietly = TRUE)) {
  cat("All required packages installed successfully!\n")
} else {
  cat("Some packages failed to install. Please check the output above.\n")
}
