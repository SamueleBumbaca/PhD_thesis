# Test simple script
library(ggplot2)

# Load data
dataset2 <- read.csv("dataset2_plots.csv")
print("Dataset2 structure:")
print(str(dataset2))
print("First few rows:")
print(head(dataset2))

# Test aggregation
treatment_avg <- aggregate(dataset2$yield, by = list(dataset2$treatment), FUN = mean)
print("Treatment averages:")
print(treatment_avg)

print("Script completed successfully!")
