# Test the corn cob emoji-based legend
# Set library path to user directory
.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# Load required packages
library(ggplot2)

# Load data
dataset2 <- read.csv("dataset2_plots.csv")

# Convert dataset2 coordinates to plot coordinates
dataset2$x_plot <- dataset2$x_center * 5
dataset2$y_plot <- dataset2$y_center * 5

# Source the functions
source("generate_trial_plots.R")

# Create a test plot with corn cob emojis
p <- ggplot() +
  geom_tile(data = dataset2, aes(x = x_plot, y = y_plot, fill = yield), 
            width = 5, height = 5, color = "white", size = 0.2) +
  scale_fill_viridis_c(name = "Yield\n(t/ha)") +
  coord_fixed(ratio = 1) +
  xlim(0, 45) + ylim(0, 30) +
  theme_minimal() +
  labs(title = "Test: Corn Cob Emojis with Integrated Legend",
       x = "X coordinate (m)", y = "Y coordinate (m)")

# Add corn cob emojis with integrated ggplot legend  
p <- add_wheat_spikes_with_legend(p, dataset2)

# Save the test plot
ggsave("test_corn_cob_emojis.png", plot = p, width = 12, height = 8, dpi = 300, bg = "white")

print("Test plot with corn cob emojis saved as test_corn_cob_emojis.png")
print("Corn cob emoji 🌽 should appear both on the plot and in the legend with different sizes")
