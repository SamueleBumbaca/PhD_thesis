# Install required packages for VS Code R extension
install.packages(c('languageserver', 'httpgd'), repos='https://cran.rstudio.com/', dependencies=TRUE)

# Check if packages are installed
if(require("languageserver", quietly = TRUE) && require("httpgd", quietly = TRUE)) {
  cat("All required packages installed successfully!\n")
} else {
  cat("Some packages failed to install. Please check the output above.\n")
}
