# Test script to verify the new legend functionality
library(ggplot2)
library(png)
library(grid)

# Load data
dataset2 <- read.csv("dataset2_plots.csv")

# Test the add_spike_legend function
source("generate_trial_plots.R")

# Create a simple test plot
p <- ggplot(dataset2, aes(x = x_center * 5, y = y_center * 5, fill = yield)) +
  geom_tile(width = 5, height = 5, color = "white", size = 0.2) +
  scale_fill_viridis_c(name = "Yield\n(t/ha)") +
  coord_fixed(ratio = 1) +
  theme_minimal() +
  labs(title = "Test Plot with New Legend")

# Add wheat spikes and the new legend
p <- add_wheat_spikes(p, dataset2)
p <- add_spike_legend(p, dataset2)

# Save the test plot
ggsave("test_legend_plot.png", plot = p, width = 12, height = 8, dpi = 300, bg = "white")

print("Test plot with new legend saved as test_legend_plot.png")
