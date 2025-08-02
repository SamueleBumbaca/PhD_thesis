# R Script to Generate Trial Data for Geostatistical Analysis Comparison
# Author: Samuele Bumbaca
# Date: July 31, 2025

# Load required libraries
library(SpATS)
library(gstat)
library(sp)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Define trial parameters
n_rows <- 6
n_cols <- 9
n_treatments <- 3
n_blocks <- 3

# Treatment effects (additive)
treatment_effects <- c(Control = 0, Test = 2, Reference = 1)

# Spatial gradient parameters
spatial_gradient_range <- c(-1.5, 1.5)  # tons/ha
base_yield <- 12  # tons/ha (baseline yield)

# Create spatial coordinates (regular grid)
x_coords <- rep(1:n_cols, n_rows)
y_coords <- rep(1:n_rows, each = n_cols)
coordinates <- data.frame(x = x_coords, y = y_coords)

# Create spatial gradient (diagonal linear trend from bottom-left to top-right)
# This creates a true diagonal gradient across both columns and rows
spatial_gradient <- with(coordinates, 
  spatial_gradient_range[1] + 
  (spatial_gradient_range[2] - spatial_gradient_range[1]) * 
  sqrt(((x - 1) / (n_cols - 1))^2 + ((y - 1) / (n_rows - 1))^2) / sqrt(2)
)

# Assign treatments to plots (3x3 blocks, each block has 5x3.33 cells approximately)
# Block 1: rows 1-2, columns 1-15
# Block 2: rows 3-4 columns 1-15  
# Block 3: rows 5-6, columns 1-15

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
  # Each treatment gets 5 columns
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

# Create Dataset 2 (plot-level averages)
# Define plot boundaries
plot_assignments <- function(x, y) {
  # 9 plots total (3 blocks × 3 treatments)
  # Each plot is approximately 5×3.33 grid cells
  
  # Determine block
  if (y <= 3) {
    block <- 1
    plot_row <- 1
  } else if (y <= 6) {
    block <- 2
    plot_row <- 2
  } else {
    block <- 3
    plot_row <- 3
  }
  
  # Determine treatment column
  if (x <= 5) {
    treatment <- "Control"
    plot_col <- 1
  } else if (x <= 10) {
    treatment <- "Test"
    plot_col <- 2
  } else {
    treatment <- "Reference"
    plot_col <- 3
  }
  
  plot_id <- paste0("B", block, "_", treatment)
  return(list(plot_id = plot_id, treatment = treatment, block = block))
}

# Apply plot assignment
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

# Fit models and extract parameters

# 1. RCBD Model (Dataset 2)
rcbd_model <- lm(yield ~ treatment + factor(block), data = dataset2)
rcbd_summary <- summary(rcbd_model)

# 2. Variogram Model (Dataset 1)
# Convert to spatial object
coordinates(dataset1) <- ~x+y

# Fit variogram on control plots only to avoid treatment effects
control_data <- dataset1[dataset1$treatment == "Control", ]
vgm_empirical <- variogram(yield ~ 1, control_data)
vgm_fit <- fit.variogram(vgm_empirical, vgm("Lin"))  # Linear model

# Spatial model with variogram
spatial_model <- gstat(formula = yield ~ treatment, data = dataset1, model = vgm_fit)

# 3. SpATS Model (Dataset 1)
# Convert back to data frame for SpATS
dataset1_df <- as.data.frame(dataset1)
spats_model <- SpATS(
  response = "yield",
  spatial = ~ SAP(x, y, nseg = c(5, 5)),
  genotype = "treatment",
  data = dataset1_df
)

# Extract model parameters for presentation
cat("=== MODEL RESULTS FOR PRESENTATION ===\n\n")

cat("1. RCBD MODEL RESULTS:\n")
cat("Treatment Effects:\n")
print(coef(rcbd_model))
cat("ANOVA Table:\n")
print(anova(rcbd_model))
cat("R-squared:", summary(rcbd_model)$r.squared, "\n")
cat("Residual SE:", summary(rcbd_model)$sigma, "\n\n")

cat("2. VARIOGRAM MODEL RESULTS:\n")
cat("Variogram parameters:\n")
print(vgm_fit)
cat("Treatment Effects (will be similar to RCBD but with spatial correction):\n\n")

cat("3. SpATS MODEL RESULTS:\n")
cat("Treatment Effects:\n")
print(spats_model$coeff[grep("treatment", names(spats_model$coeff))])
cat("Spatial variance explained:", spats_model$psi[1], "\n")
cat("Error variance:", spats_model$psi[2], "\n\n")

# Save datasets
write.csv(dataset1_df, "dataset1_detailed.csv", row.names = FALSE)
write.csv(dataset2, "dataset2_plots.csv", row.names = FALSE)

# Save model objects
save(rcbd_model, vgm_fit, spats_model, dataset1_df, dataset2, 
     file = "trial_models.RData")

cat("Data and models saved successfully!\n")
cat("Files created:\n")
cat("- dataset1_detailed.csv (detailed grid data)\n")
cat("- dataset2_plots.csv (plot-level averages)\n") 
cat("- trial_models.RData (fitted models)\n")

# Print some summary statistics for the presentation
cat("\n=== SUMMARY STATISTICS FOR SLIDES ===\n")
cat("Yield range:", round(range(dataset1_df$yield), 2), "tons/ha\n")
cat("Spatial gradient range:", round(range(dataset1_df$spatial_gradient), 2), "tons/ha\n")
cat("True treatment effects:", treatment_effects, "tons/ha\n")
cat("Number of observations per plot:", round(mean(dataset2$n_points)), "\n")
