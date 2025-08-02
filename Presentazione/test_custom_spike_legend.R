# Test the new wheat spike legend approach
# Set library path to user directory
.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# Load required packages
library(ggplot2)
library(png)
library(grid)

# Load data
dataset2 <- read.csv("dataset2_plots.csv")

# Load the wheat spike icon
spiga_img <- readPNG("Imgs/Spiga.png")

# Convert dataset2 coordinates to plot coordinates
dataset2$x_plot <- dataset2$x_center * 5
dataset2$y_plot <- dataset2$y_center * 5

# Source the functions
source("generate_trial_plots.R")

# Create a simple test plot
p <- ggplot() +
  geom_tile(data = dataset2, aes(x = x_plot, y = y_plot, fill = yield), 
            width = 5, height = 5, color = "white", size = 0.2) +
  scale_fill_viridis_c(name = "Yield\n(t/ha)") +
  coord_fixed(ratio = 1) +
  xlim(0, 45) + ylim(0, 30) +
  theme_minimal() +
  labs(title = "Test Plot with Custom Wheat Spike Legend",
       x = "X coordinate (m)", y = "Y coordinate (m)")

# Add wheat spikes with custom spike legend
p <- add_wheat_spikes_with_legend(p, dataset2)

# Save the test plot
ggsave("test_custom_spike_legend.png", plot = p, width = 12, height = 8, dpi = 300, bg = "white")

print("Test plot with custom spike legend saved as test_custom_spike_legend.png")
print(paste("Yield range:", min(dataset2$yield), "to", max(dataset2$yield)))
