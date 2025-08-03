# Load packages
library(ggplot2)

# Read the data
dataset1 <- read.csv("dataset1_detailed.csv")
dataset2 <- read.csv("dataset2_plots.csv")

# Settings
plot_width <- 12
plot_height <- 8
dpi <- 300

# Convert dataset2 coordinates to plot coordinates (scale from grid to field coordinates)
# Original grid: 9 cols x 6 rows -> Field: 45m x 30m
dataset2$x_plot <- dataset2$x_center * 5  # Scale x: 9 grid units -> 45m
dataset2$y_plot <- dataset2$y_center * 5  # Scale y: 6 grid units -> 30m

# Function to add corn cobs with integrated ggplot legend using individual observations
add_wheat_spikes_with_legend <- function(plot_obj, data) {
  # Use individual observations data directly
  individual_data <- data
  
  # Define yield categories based on quantiles of individual yields
  individual_data$yield_category <- cut(individual_data$yield, 
                                       breaks = c(-Inf, quantile(individual_data$yield, 0.33), 
                                                 quantile(individual_data$yield, 0.67), Inf),
                                       labels = c("Low", "Medium", "High"))
  
  # Define corresponding sizes for each category
  size_mapping <- c("Low" = 3, "Medium" = 5, "High" = 7)
  individual_data$spike_size <- size_mapping[individual_data$yield_category]
  
  # Add wheat spike emoji annotations for each individual observation
  for(i in 1:nrow(individual_data)) {
    plot_obj <- plot_obj + 
      annotate("text", x = individual_data$x[i], y = individual_data$y[i], 
               label = "🌽", size = individual_data$spike_size[i], hjust = 0.5, vjust = 0.5)
  }
  
  # Add treatment letters at plot centers
  for(i in 1:nrow(plot_coords)) {
    treatment_letter <- switch(plot_coords$treatment[i],
                              "T" = "T",
                              "C" = "C", 
                              "R" = "R")
    
    plot_obj <- plot_obj + 
      annotate("text", x = plot_coords$x_center[i], y = plot_coords$y_center[i], 
               label = treatment_letter, size = 13, hjust = 0.5, vjust = 0.5, color = "black")
  }
  
  # Create invisible points for ggplot legend with corn cob emoji
  individual_data$yield_category <- factor(individual_data$yield_category, levels = c("Low", "Medium", "High"))
  
  # Get yield range for legend labels
  yield_range <- range(individual_data$yield)
  
  # Get the actual levels present in the data
  present_levels <- levels(individual_data$yield_category)[levels(individual_data$yield_category) %in% individual_data$yield_category]
  n_levels <- length(present_levels)
  
  plot_obj <- plot_obj + 
    geom_point(data = individual_data, 
               aes(x = x, y = y, size = yield_category),
               shape = 15, color = "wheat4", alpha = 0) +  # Invisible base points
    scale_size_manual(
      name = "Corn Cobs\nYield (t/ha)", 
      values = c("Low" = 3, "Medium" = 5, "High" = 7),  # Sizes for legend
      labels = c("Low" = sprintf("%.1f", quantile(individual_data$yield, 0.17)),
                "Medium" = sprintf("%.1f", quantile(individual_data$yield, 0.50)),
                "High" = sprintf("%.1f", quantile(individual_data$yield, 0.83))),
      guide = guide_legend(
        override.aes = list(
          alpha = 1,
          shape = rep("🌽", n_levels),  # Use the correct number of emoji symbols
          color = c("gold2", "gold3", "gold4")[1:n_levels],  # Use only the needed colors
          size = c(4, 6, 8)[1:n_levels]  # Use only the needed sizes
        ),
        title.position = "top",
        title.hjust = 0.5,
        label.position = "right",
        keywidth = unit(1.2, "cm"),
        keyheight = unit(1, "cm"),
        title.theme = element_text(size = 12, face = "bold"),
        label.theme = element_text(size = 10)
      )
    )
  
  return(plot_obj)
}

# Define plot coordinates and treatments
plot_coords <- data.frame(
  plot_id = 1:9,
  x_center = rep(c(7.5, 22.5, 37.5), 3),
  y_center = rep(c(25, 15, 5), each = 3),
  treatment = c("T", "C", "R",
                "R", "T", "C", 
                "C", "R", "T"),
  block = rep(c("Block 1", "Block 2", "Block 3"), each = 3)
)

# Color palettes
treatment_colors <- c("Control" = "#DCDCDC", "Test" = "#ADD8E6", "Reference" = "#FFB6C1")
block_colors <- c("Block 1" = "#d3ff6b", "Block 2" = "#5b4ecd", "Block 3" = "#b045d1")

# Create RCBD plot
create_rcbd_plot <- function() {
  # Create rectangles for each plot area with gaps to prevent overlap
  rectangles <- data.frame()
  for (i in 1:nrow(plot_coords)) {
    rect_data <- data.frame(
      x = c(rep(plot_coords$x_center[i] - 7.4, 2), rep(plot_coords$x_center[i] + 7.4, 2)),
      y = c(plot_coords$y_center[i] - 4.9, plot_coords$y_center[i] + 4.9, 
            plot_coords$y_center[i] + 4.9, plot_coords$y_center[i] - 4.9),
      plot_id = plot_coords$plot_id[i],
      treatment = plot_coords$treatment[i],
      block = plot_coords$block[i]
    )
    rectangles <- rbind(rectangles, rect_data)
  }
  
  # Calculate true_treatment_effect values for border coloring
  rectangles$true_treatment_effect <- NA
  for(i in 1:nrow(rectangles)) {
    rectangles$true_treatment_effect[i] <- switch(rectangles$treatment[i],
                                                  "C" = 0,    # Control
                                                  "T" = 1,    # Test  
                                                  "R" = 0.5)  # Reference
  }
  
  p <- ggplot() +
    # Add polygons for each plot with colored borders
    geom_polygon(data = rectangles, 
                 aes(x = x, y = y, group = plot_id, color = true_treatment_effect), 
                 fill = NA, size = 3) +
    scale_color_gradient2(
      name = "Treatment Effect\n(True Value)", 
      low = "lightblue", mid = "white", high = "lightcoral",
      midpoint = 0.5, guide = "colorbar"
    ) +
    # Add block color as background
    geom_rect(data = plot_coords, 
              aes(xmin = x_center - 7.4, xmax = x_center + 7.4, 
                  ymin = y_center - 4.9, ymax = y_center + 4.9, fill = block), 
              alpha = 0.3, color = NA) +
    scale_fill_manual(name = "Blocks", values = block_colors) +
    coord_fixed(ratio = 1) +
    labs(
      title = "RCBD (Randomized Complete Block Design) - Trial Design",
      x = "X Coordinate (m)",
      y = "Y Coordinate (m)"
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", size = 0.5),
      panel.grid.minor = element_line(color = "gray95", size = 0.3),
      axis.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 12),
      legend.position = "right",
      legend.box = "vertical",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  # Add corn cobs with ggplot legend
  p <- add_wheat_spikes_with_legend(p, dataset1)
  
  return(p)
}

# Generate RCBD plot
cat("Generating RCBD trial design plot...\n")
rcbd_plot <- create_rcbd_plot()
ggsave("Imgs/rcbd_trial_design_blocks.png", rcbd_plot, 
       width = plot_width, height = plot_height, dpi = dpi)

cat("RCBD plot generated successfully!\n")
