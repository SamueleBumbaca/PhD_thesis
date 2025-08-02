# Generate Trial Design Plots for Presentation
# This script creates PNG files showing trial design with environmental gradients
# and model-specific effects overlaid in different colors

# Load required packages
library(ggplot2)
library(dplyr)
library(gridExtra)
library(metR)  # For geom_text_contour

# Set up plotting parameters
plot_width <- 12
plot_height <- 8
dpi <- 300

# Create directory for images if it doesn't exist
if (!dir.exists("Imgs")) {
  dir.create("Imgs")
}

# Load the generated data and models
load("trial_models.RData")
dataset1 <- read.csv("dataset1_detailed.csv")
dataset2 <- read.csv("dataset2_plots.csv")

# Define plot coordinates and treatments
plot_coords <- data.frame(
  plot_id = 1:9,
  x_center = rep(c(7.5, 22.5, 37.5), 3),
  y_center = rep(c(25, 15, 5), each = 3),
  treatment = c("Test", "Control", "Reference",
                "Reference", "Test", "Control", 
                "Control", "Reference", "Test"),
  block = rep(c("Block 1", "Block 2", "Block 3"), each = 3)
)

# Color palettes
treatment_colors <- c("Control" = "#DCDCDC", "Test" = "#ADD8E6", "Reference" = "#FFB6C1")
block_colors <- c("Block 1" = "#d3ff6b", "Block 2" = "#5b4ecd", "Block 3" = "#b045d1")

# Base plot function
create_base_plot <- function() {
  ggplot() +
    # Plot boundaries
    geom_rect(data = plot_coords, 
              aes(xmin = x_center - 7.5, xmax = x_center + 7.5,
                  ymin = y_center - 5, ymax = y_center + 5,
                  fill = treatment), 
              color = "black", alpha = 0.3, size = 1) +
    scale_fill_manual(values = treatment_colors, name = "Treatment") +
    # Plot labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 4, fontface = "bold") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "#000000", size = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
}

# 1. RCBD Trial Design with Block Effects
create_rcbd_plot <- function() {
  # Environmental gradient (true effect) - use actual diagonal gradient from data
  x_seq <- seq(0, 45, length.out = 100)
  y_seq <- seq(0, 30, length.out = 100)
  grid_data <- expand.grid(x = x_seq, y = y_seq)
  
  # Diagonal gradient: bottom-left to top-right (matching data generation)
  # Scale to field coordinates: x=0-45m, y=0-30m
  x_norm <- grid_data$x / 45  # 0 to 1
  y_norm <- grid_data$y / 30  # 0 to 1
  diagonal_dist <- sqrt(x_norm^2 + y_norm^2) / sqrt(2)  # 0 to 1
  grid_data$env_gradient <- -1.5 + 3 * diagonal_dist  # -1.5 to +1.5 t/ha
  
  # Block effects (discrete)
  plot_coords$block_effect <- c(0.47, 0.47, 0.47,  # Block 1
                                0.91, 0.91, 0.91,  # Block 2  
                                0, 0, 0)            # Block 3 (reference)
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "True Gradient\n(t/ha)") +
    # Block effect overlay
    geom_rect(data = plot_coords, 
              aes(xmin = x_center - 7.5, xmax = x_center + 7.5,
                  ymin = y_center - 5, ymax = y_center + 5,
                  color = block), 
              fill = NA, size = 3, alpha = 0.8) +
    scale_color_manual(values = block_colors, name = "Block Effect") +
    # Treatment labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 4, fontface = "bold", color = "white") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "RCBD: Environmental Gradient vs Block Effects") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", size = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  return(p)
}

# 2. Variogram Trial Design with Spatial Effects
create_variogram_plot <- function() {
  # Environmental gradient (true effect) - use actual diagonal gradient from data
  x_seq <- seq(0, 45, length.out = 100)
  y_seq <- seq(0, 30, length.out = 100)
  grid_data <- expand.grid(x = x_seq, y = y_seq)
  
  # Diagonal gradient: bottom-left to top-right (matching data generation)
  x_norm <- grid_data$x / 45  # 0 to 1
  y_norm <- grid_data$y / 30  # 0 to 1
  diagonal_dist <- sqrt(x_norm^2 + y_norm^2) / sqrt(2)  # 0 to 1
  grid_data$env_gradient <- -1.5 + 3 * diagonal_dist  # -1.5 to +1.5 t/ha
  
  # Simulated spatial effect from variogram (diagonal trend with some variation)
  grid_data$spatial_effect <- -1.2 + 2.4 * diagonal_dist + 
                              0.3 * sin(2 * pi * grid_data$y / 30)
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "True Gradient\n(t/ha)") +
    # Spatial effect contours
    geom_contour(data = grid_data, aes(x = x, y = y, z = spatial_effect), 
                color = "orange", size = 1.5, alpha = 0.8) +
    # Spatial effect contour labels
    geom_text_contour(data = grid_data, aes(x = x, y = y, z = spatial_effect),
                     color = "darkorange", size = 3, fontface = "bold") +
    # Treatment boundaries
    geom_rect(data = plot_coords, 
              aes(xmin = x_center - 7.5, xmax = x_center + 7.5,
                  ymin = y_center - 5, ymax = y_center + 5), 
              fill = NA, color = "black", size = 1) +
    # Treatment labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 4, fontface = "bold", color = "white") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "Variogram: Environmental Gradient vs Spatial Effects") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", size = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    ) +
    # Add legend for spatial effect
    annotate("text", x = 35, y = 28, label = "Orange contours:\nSpatial effects", 
             size = 4, hjust = 0, color = "orange", fontface = "bold")
  
  return(p)
}

# 3. SpATS Trial Design with Spline Effects
create_spats_plot <- function() {
  # Environmental gradient (true effect) - use actual diagonal gradient from data
  x_seq <- seq(0, 45, length.out = 100)
  y_seq <- seq(0, 30, length.out = 100)
  grid_data <- expand.grid(x = x_seq, y = y_seq)
  
  # Diagonal gradient: bottom-left to top-right (matching data generation)
  x_norm <- grid_data$x / 45  # 0 to 1
  y_norm <- grid_data$y / 30  # 0 to 1
  diagonal_dist <- sqrt(x_norm^2 + y_norm^2) / sqrt(2)  # 0 to 1
  grid_data$env_gradient <- -1.5 + 3 * diagonal_dist  # -1.5 to +1.5 t/ha
  
  # Simulated spline effect (smooth spatial surface following diagonal pattern)
  grid_data$spline_effect <- -1.0 + 2.0 * diagonal_dist + 
                            0.5 * sin(2 * pi * diagonal_dist) +
                            0.2 * cos(2 * pi * grid_data$y / 30)
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "True Gradient\n(t/ha)") +
    # Spline effect contours
    geom_contour(data = grid_data, aes(x = x, y = y, z = spline_effect), 
                color = "green", size = 1.5, alpha = 0.8) +
    # Spline effect contour labels
    geom_text_contour(data = grid_data, aes(x = x, y = y, z = spline_effect),
                     color = "darkgreen", size = 3, fontface = "bold") +
    # Treatment boundaries
    geom_rect(data = plot_coords, 
              aes(xmin = x_center - 7.5, xmax = x_center + 7.5,
                  ymin = y_center - 5, ymax = y_center + 5), 
              fill = NA, color = "black", size = 1) +
    # Treatment labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 4, fontface = "bold", color = "white") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "SpATS: Environmental Gradient vs Spline Effects") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", size = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    ) +
    # Add legend for spline effect
    annotate("text", x = 35, y = 28, label = "Green contours:\nSpline effects", 
             size = 4, hjust = 0, color = "darkgreen", fontface = "bold")
  
  return(p)
}

# Generate and save plots
cat("Generating RCBD trial design plot...\n")
rcbd_plot <- create_rcbd_plot()
ggsave("Imgs/rcbd_trial_design_blocks.png", rcbd_plot, 
       width = plot_width, height = plot_height, dpi = dpi)

cat("Generating Variogram trial design plot...\n")
variogram_plot <- create_variogram_plot()
ggsave("Imgs/variogram_trial_design_spatial.png", variogram_plot, 
       width = plot_width, height = plot_height, dpi = dpi)

cat("Generating SpATS trial design plot...\n")
spats_plot <- create_spats_plot()
ggsave("Imgs/spats_trial_design_spline.png", spats_plot, 
       width = plot_width, height = plot_height, dpi = dpi)

cat("All plots generated successfully!\n")
cat("Files created:\n")
cat("- Imgs/rcbd_trial_design_blocks.png\n")
cat("- Imgs/variogram_trial_design_spatial.png\n")
cat("- Imgs/spats_trial_design_spline.png\n")
