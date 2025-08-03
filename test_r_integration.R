# Test R script for VS Code integration
# This script tests basic R functionality

# Print R version
cat("R version:", R.version.string, "\n")

# Test basic operations
x <- 1:10
y <- x^2
plot(x, y, main="Simple Plot", xlab="X values", ylab="Y values")

# Check if languageserver is available
if (requireNamespace("languageserver", quietly = TRUE)) {
  cat("languageserver package is available\n")
  cat("You should now be able to use .vsc.attach()\n")
} else {
  cat("languageserver package is NOT available\n")
}

# Test .vsc.attach() function
# This function should work now
cat("Testing .vsc.attach()...\n")
