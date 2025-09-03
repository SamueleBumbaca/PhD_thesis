# Install required packages
.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# List of required packages
required_packages <- c(
  "SpATS", "gstat", "sp", "dplyr", "ggplot2", "metR", "gridExtra",
  "lme4", "car", "lmtest", "spdep", "scales"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing package:", pkg, "\n")
    install.packages(pkg, repos = "https://cran.r-project.org/")
    library(pkg, character.only = TRUE)
  } else {
    cat("Package", pkg, "already installed\n")
  }
}

cat("All required packages are available\n")
