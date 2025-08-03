# Simplified data generation script
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Define trial parameters
n_rows <- 6
n_cols <- 9
n_treatments <- 3
n_blocks <- 3

# NEW Treatment effects (additive)
treatment_effects <- c(Control = 0, Test = 1, Reference = 0.5)

# Spatial gradient parameters
spatial_gradient_range <- c(-1.5, 1.5)  # tons/ha
base_yield <- 12  # tons/ha (baseline yield)

# Create spatial coordinates (regular grid)
x_coords <- rep(1:n_cols, n_rows)
y_coords <- rep(1:n_rows, each = n_cols)
coordinates <- data.frame(x = x_coords, y = y_coords)

# Create spatial gradient (diagonal linear trend from bottom-left to top-right)
spatial_gradient <- with(coordinates, 
  spatial_gradient_range[1] + 
  (spatial_gradient_range[2] - spatial_gradient_range[1]) * 
  sqrt(((x - 1) / (n_cols - 1))^2 + ((y - 1) / (n_rows - 1))^2) / sqrt(2)
)

# Assign treatments to plots
create_treatment_assignment <- function(x, y) {
  # Determine block based on row
  if (y <= 2) {
    block <- 1
  } else if (y <= 4) {
    block <- 2
  } else {
    block <- 3
  }
  
  # Within each block, assign treatments in a systematic pattern
  if (x <= 3) {
    treatment <- "Control"
  } else if (x <= 6) {
    treatment <- "Test"
  } else {
    treatment <- "Reference"
  }
  
  return(list(treatment = treatment, block = block))
}

# Apply treatment assignment
assignments <- mapply(create_treatment_assignment, coordinates$x, coordinates$y, SIMPLIFY = FALSE)
treatments <- sapply(assignments, function(x) x$treatment)
blocks <- sapply(assignments, function(x) x$block)

# Calculate treatment effects
treatment_effect_values <- treatment_effects[treatments]

# Add random error
random_error <- rnorm(nrow(coordinates), mean = 0, sd = 0.3)

# Calculate final yield values
yield_values <- base_yield + spatial_gradient + treatment_effect_values + random_error

# Ensure yield values are within the specified range
yield_values <- pmax(9.6, pmin(14.2, yield_values))

# Create Dataset 1 (detailed grid data)
dataset1 <- data.frame(
  x = coordinates$x,
  y = coordinates$y,
  treatment = treatments,
  block = blocks,
  spatial_gradient = spatial_gradient,
  treatment_effect = treatment_effect_values,
  yield = yield_values
)

# Add plot_id
plot_assignments <- function(x, y) {
  # Determine block
  if (y <= 3) {
    block <- 1
  } else if (y <= 6) {
    block <- 2
  } else {
    block <- 3
  }
  
  # Determine treatment column
  if (x <= 5) {
    treatment <- "Control"
  } else if (x <= 10) {
    treatment <- "Test"
  } else {
    treatment <- "Reference"
  }
  
  plot_id <- paste0("B", block, "_", treatment)
  return(list(plot_id = plot_id, treatment = treatment, block = block))
}

plot_info <- mapply(plot_assignments, coordinates$x, coordinates$y, SIMPLIFY = FALSE)
dataset1$plot_id <- sapply(plot_info, function(x) x$plot_id)

# Calculate plot-level averages for Dataset 2
dataset2 <- dataset1 %>%
  group_by(plot_id, treatment, block) %>%
  summarise(
    x_center = mean(x),
    y_center = mean(y),
    yield = mean(yield),
    true_spatial_effect = mean(spatial_gradient),
    true_treatment_effect = mean(treatment_effect),
    n_points = n(),
    .groups = 'drop'
  )

# Save datasets
write.csv(dataset1, "dataset1_detailed.csv", row.names = FALSE)
write.csv(dataset2, "dataset2_plots.csv", row.names = FALSE)

cat("Data generated successfully with new treatment effects!\n")
cat("Treatment effects: Control =", treatment_effects["Control"], 
    ", Test =", treatment_effects["Test"], 
    ", Reference =", treatment_effects["Reference"], "\n")
cat("Files created:\n")
cat("- dataset1_detailed.csv (detailed grid data)\n")
cat("- dataset2_plots.csv (plot-level averages)\n")
