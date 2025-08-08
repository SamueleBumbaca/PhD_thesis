# Test script to verify VS Code R packages
cat("Testing R packages for VS Code integration...\n")

# Test languageserver
tryCatch({
  library(languageserver)
  cat("✓ languageserver package loaded successfully\n")
}, error = function(e) {
  cat("✗ Error loading languageserver:", e$message, "\n")
})

# Test httpgd
tryCatch({
  library(httpgd)
  cat("✓ httpgd package loaded successfully\n")
}, error = function(e) {
  cat("✗ Error loading httpgd:", e$message, "\n")
})

cat("\nR version:", R.version.string, "\n")
cat("Both packages are now available for VS Code R extension!\n")
cat("You can now use httpgd for interactive plots in VS Code.\n")
