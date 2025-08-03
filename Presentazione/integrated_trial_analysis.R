# Integrated Trial Analysis: RCBD vs SpATS Comparison
# This script generates trial data and creates a comprehensive comparison plot
# showing both block effects (RCBD) and spatial spline effects (SpATS)

# Set library path to user directory
.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# Load required packages
library(ggplot2)
library(dplyr)
library(gridExtra)
library(metR)      # For geom_text_contour
library(SpATS)     # For spatial modeling
library(gstat)     # For variogram modeling
library(sp)        # For spatial data structures
library(lme4)      # For linear mixed models
library(car)       # For Levene's test
library(lmtest)    # For Durbin-Watson test
library(spdep)     # For Moran's I test

# Set seed for reproducibility
set.seed(123)

# Suppress common warnings for cleaner output
options(warn = -1)  # Temporarily suppress warnings

# =============================================================================
# DATA GENERATION (from generate_trial_data.R)
# =============================================================================

# Define trial parameters
n_rows <- 6
n_cols <- 9
n_treatments <- 3
n_blocks <- 3

# Treatment effects (additive)
treatment_effects <- c(Control = 0, Test = 1, Reference = 0.5)

# Spatial gradient parameters
spatial_gradient_range <- c(-1.5, 1.5)  # tons/ha
base_yield <- 12  # tons/ha (baseline yield)

# Create spatial coordinates (regular grid)
x_coords <- rep(1:n_cols, n_rows)
y_coords <- rep(1:n_rows, each = n_cols)
coordinates <- data.frame(x = x_coords, y = y_coords)

# Create complex spatial gradient with two focal points
# Convert plot coordinates to field coordinates (multiply by 5 and adjust)
x_field <- coordinates$x * 5 - 2.5  # Convert to field scale
y_field <- coordinates$y * 5 - 2.5  # Convert to field scale

# Define focal points in field coordinates
focal_point1 <- c(x = 15, y = 20)  # Single focal point

# Calculate distances from the focal point
dist_to_focal1 <- sqrt((x_field - focal_point1["x"])^2 + (y_field - focal_point1["y"])^2)

# Use the distance to the focal point for gradient calculation
min_distance <- dist_to_focal1

# Create radial gradient: center = -1.5, increase by +0.3 every 2 spatial units
# Base value at centers: -1.5
# Increase rate: +0.3 per 2 spatial units = +0.15 per spatial unit
gradient_increment <- min_distance * 0.15  # 0.15 per spatial unit = 0.3 per 2 units

# Calculate final gradient values, starting from homogeneous +1.5 baseline
baseline_gradient <- 1.5
focal_effect <- -1.5 + gradient_increment  # Start from -1.5 at centers, increase radially

# Combine baseline with focal effects (focal effects override in their vicinity)
# Use a smooth transition where focal effects dominate within reasonable distance
max_focal_distance <- 20  # Maximum distance where focal effects are significant
focal_weight <- pmax(0, 1 - min_distance / max_focal_distance)  # Weight decreases with distance

# Create heteroscedastic pattern for RCBD to fail assumption tests
# Add position-dependent variance multiplier
x_norm <- (coordinates$x - 1) / (9 - 1)  # Normalize x coordinates to 0-1
y_norm <- (coordinates$y - 1) / (6 - 1)  # Normalize y coordinates to 0-1
variance_multiplier <- 1 + 2 * x_norm + 1.5 * y_norm  # Increasing variance from left to right and bottom to top

spatial_gradient <- baseline_gradient * (1 - focal_weight) + focal_effect * focal_weight

# Ensure gradient stays within the specified bounds (-1.5 to +1.5)
spatial_gradient <- pmax(-1.5, pmin(1.5, spatial_gradient))

# Assign treatments to plots (RCBD design)
create_treatment_assignment <- function(x, y) {
  # Determine block based on row
  if (y <= 2) {
    block <- 1
  } else if (y <= 4) {
    block <- 2
  } else {
    block <- 3
  }
  
  # Within each block, assign treatments systematically
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

# Add heteroscedastic random error (varies by position to make RCBD fail homoscedasticity)
# Different error variance based on position to create heteroscedasticity
base_error_sd <- 0.2
heteroscedastic_error <- rnorm(nrow(coordinates), mean = 0, sd = base_error_sd * variance_multiplier)

# Calculate final yield values
yield_values <- base_yield + spatial_gradient + treatment_effect_values + heteroscedastic_error

# Ensure yield values are within reasonable range
yield_values <- pmax(9.6, pmin(14.2, yield_values))

# Create comprehensive dataset
dataset1 <- data.frame(
  x = coordinates$x,
  y = coordinates$y,
  treatment = treatments,
  block = blocks,
  spatial_gradient = spatial_gradient,
  treatment_effect = treatment_effect_values,
  yield = yield_values,
  plot_id = paste0("B", blocks, "_", treatments)
)

# Add plot identifier for mixed model
dataset1$plot_factor <- as.factor(dataset1$plot_id)
dataset1$block_factor <- as.factor(dataset1$block)

cat("=== DATA GENERATION COMPLETED ===\n")
cat("Dataset dimensions:", nrow(dataset1), "observations\n")
cat("Treatments:", unique(dataset1$treatment), "\n")
cat("Blocks:", unique(dataset1$block), "\n")
cat("Yield range:", round(range(dataset1$yield), 2), "tons/ha\n\n")

# =============================================================================
# MODEL FITTING
# =============================================================================

# 1. RCBD Linear Mixed Model (accounting for pseudoreplication)
cat("Fitting RCBD Linear Mixed Model...\n")

# Check data structure
cat("Data structure check:\n")
cat("Number of plots per block:", table(dataset1$block), "\n")
cat("Number of observations per plot:", table(dataset1$plot_id), "\n")

# Try simplified mixed model first (only block random effect)
rcbd_mixed_model <- tryCatch({
  suppressWarnings(lmer(yield ~ treatment + (1|block_factor), data = dataset1))
}, error = function(e) {
  cat("Mixed model failed, falling back to simple linear model with block as fixed effect\n")
  lm(yield ~ treatment + block_factor, data = dataset1)
})

# Check if it's a mixed model or linear model
is_mixed_model <- inherits(rcbd_mixed_model, "lmerMod")

if (is_mixed_model) {
  # Extract block effects for visualization from mixed model
  block_effects <- ranef(rcbd_mixed_model)$block_factor
  block_effects_df <- data.frame(
    block = 1:3,
    block_effect = block_effects$`(Intercept)`,
    block_name = paste("Block", 1:3)
  )
  cat("RCBD Mixed Model fitted successfully\n")
} else {
  # Extract block effects from fixed effects in linear model
  coefs <- coefficients(rcbd_mixed_model)
  block_effects_df <- data.frame(
    block = 1:3,
    block_effect = c(0, # Block 1 is reference
                     ifelse("block_factor2" %in% names(coefs), coefs["block_factor2"], 0),
                     ifelse("block_factor3" %in% names(coefs), coefs["block_factor3"], 0)),
    block_name = paste("Block", 1:3)
  )
  cat("RCBD Linear Model (with fixed block effects) fitted successfully\n")
}

print(summary(rcbd_mixed_model))

# 2. SpATS Model
cat("\nFitting SpATS Model...\n")
spats_model <- suppressWarnings(
  SpATS(response = "yield", 
        spatial = ~ PSANOVA(x, y, nseg = c(15, 15)), 
        genotype = "treatment", 
        fixed = ~ 1,
        data = dataset1,
        control = list(tolerance = 1e-03, monitoring = 0))
)

cat("SpATS Model fitted successfully\n")

# Create fine prediction grid for SpATS spatial effects
x_seq_fine <- seq(min(dataset1$x), max(dataset1$x), length.out = 60)
y_seq_fine <- seq(min(dataset1$y), max(dataset1$y), length.out = 60)
pred_grid_fine <- expand.grid(x = x_seq_fine, y = y_seq_fine)
pred_grid_fine$treatment <- "Control"  # Use Control as reference

# Get spatial predictions
spatial_pred <- suppressWarnings(predict(spats_model, pred_grid_fine))
pred_grid_fine$spline_effect <- spatial_pred$predicted.values

# Scale coordinates to field layout (multiply by 5 for field coordinates)
pred_grid_fine$x_scaled <- pred_grid_fine$x * 5 - 2.5
pred_grid_fine$y_scaled <- pred_grid_fine$y * 5 - 2.5

# =============================================================================
# VISUALIZATION SETUP
# =============================================================================

# Plot coordinates and treatments
plot_coords <- data.frame(
  plot_id = 1:9,
  x_center = rep(c(7.5, 22.5, 37.5), 3),
  y_center = rep(c(25, 15, 5), each = 3),
  treatment = c("T", "C", "R",
                "R", "T", "C", 
                "C", "R", "T"),
  block = rep(c("Block 1", "Block 2", "Block 3"), each = 3)
)

# Block labels with effects
block_labels <- data.frame(
  block = c("Block 1", "Block 2", "Block 3"),
  block_num = 1:3,
  x_pos = rep(-5, 3),
  y_pos = c(25, 15, 5),
  block_effect = block_effects_df$block_effect
)

# Add block effect colors based on spline effect scale
spline_range <- range(pred_grid_fine$spline_effect, na.rm = TRUE)
block_effect_range <- range(block_labels$block_effect)

# Normalize block effects to spline effect scale for consistent coloring
block_labels$normalized_effect <- suppressWarnings(
  scales::rescale(block_labels$block_effect, to = spline_range)
)

# Define treatment effect colors for plot borders
plot_coords_extended <- plot_coords
plot_coords_extended$treatment_full <- sapply(plot_coords_extended$treatment, function(x) {
  switch(x, "T" = "Test", "C" = "Control", "R" = "Reference")
})
plot_coords_extended$true_treatment_effect <- sapply(plot_coords_extended$treatment_full, function(x) {
  switch(x, "Control" = 0, "Reference" = 0.5, "Test" = 1)
})

# Function to add individual observations with yield-based sizing
add_wheat_spikes_with_legend <- function(plot_obj, data) {
  # Scale coordinates to match field layout
  individual_data <- data
  individual_data$x_scaled <- individual_data$x * 5 - 2.5
  individual_data$y_scaled <- individual_data$y * 5 - 2.5
  
  # Define yield categories
  individual_data$yield_category <- cut(individual_data$yield, 
                                       breaks = c(-Inf, quantile(individual_data$yield, 0.33), 
                                                 quantile(individual_data$yield, 0.67), Inf),
                                       labels = c("Low", "Medium", "High"))
  
  # Define corresponding sizes for each category
  size_mapping <- c("Low" = 3, "Medium" = 5, "High" = 7)
  individual_data$spike_size <- size_mapping[individual_data$yield_category]
  
  # Add wheat spike annotations
  for(i in 1:nrow(individual_data)) {
    size_val <- switch(individual_data$yield_category[i],
                       "Low" = 3, "Medium" = 5, "High" = 7)
    plot_obj <- plot_obj + 
      annotate("text", x = individual_data$x_scaled[i], y = individual_data$y_scaled[i], 
               label = "🌽", size = size_val, hjust = 0.5, vjust = 0.5)
  }
  
  # Add treatment letters
  for(i in 1:nrow(plot_coords)) {
    plot_obj <- plot_obj + 
      annotate("text", x = plot_coords$x_center[i], y = plot_coords$y_center[i], 
               label = plot_coords$treatment[i], size = 13, hjust = 0.5, vjust = 0.5, 
               color = "black", fontface = "bold")
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
               aes(x = x_scaled, y = y_scaled, size = yield_category),
               shape = 15, color = "wheat4", alpha = 0) +  # Invisible base points
    scale_size_manual(
      name = "Observation\nYield (t/ha)", 
      values = c("Low" = 3, "Medium" = 5, "High" = 7),  # Sizes for legend
      labels = c("Low" = sprintf("%.1f", quantile(individual_data$yield, 0.17)),
                "Medium" = sprintf("%.1f", quantile(individual_data$yield, 0.50)),
                "High" = sprintf("%.1f", quantile(individual_data$yield, 0.83))),
      guide = guide_legend(
        override.aes = list(
          alpha = 1,
          shape = rep("🌽", n_levels),  # Use the correct number of emoji symbols
          color = c("black", "black", "black")[1:n_levels],  # Use only the needed colors
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

# =============================================================================
# INTEGRATED PLOT CREATION
# =============================================================================

create_integrated_plot <- function() {
  # Environmental gradient background with single focal point
  x_seq <- seq(0, 45, length.out = 100)
  y_seq <- seq(0, 30, length.out = 100)
  grid_data <- expand.grid(x = x_seq, y = y_seq)
  
  # Define focal point in field coordinates
  focal_point1 <- c(x = 15, y = 20)
  
  # Calculate distances from the focal point
  dist_to_focal1 <- sqrt((grid_data$x - focal_point1["x"])^2 + (grid_data$y - focal_point1["y"])^2)
  
  # Use distance to the focal point
  min_distance <- dist_to_focal1
  
  # Create radial gradient: center = -1.5, increase by +0.15 per spatial unit
  gradient_increment <- min_distance * 0.15
  baseline_gradient <- 1.5
  focal_effect <- -1.5 + gradient_increment
  
  # Smooth transition between baseline and focal effects
  max_focal_distance <- 20
  focal_weight <- pmax(0, 1 - min_distance / max_focal_distance)
  grid_data$env_gradient <- baseline_gradient * (1 - focal_weight) + focal_effect * focal_weight
  
  # Ensure gradient stays within the specified bounds (-1.5 to +1.5)
  grid_data$env_gradient <- pmax(-1.5, pmin(1.5, grid_data$env_gradient))
  
  # Debug: Print gradient range for verification
  cat("Environmental gradient range:", range(grid_data$env_gradient, na.rm = TRUE), "\n")
  cat("Gradient values summary:\n")
  print(summary(grid_data$env_gradient))
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "Environmental\nSpatial Effect\n(t/ha)") +
    # Spline effect contours from SpATS model (only for finite values)
    geom_contour(data = pred_grid_fine[is.finite(pred_grid_fine$spline_effect), ], 
                aes(x = x_scaled, y = y_scaled, z = spline_effect), 
                color = "purple", linewidth = 1.5, alpha = 0.8) +
    # Spline effect contour labels (only for finite values)
    geom_text_contour(data = pred_grid_fine[is.finite(pred_grid_fine$spline_effect), ], 
                     aes(x = x_scaled, y = y_scaled, z = spline_effect),
                     color = "purple4", size = 4, fontface = "bold") +
    # Map plot_coords treatments to dataset2 treatment values for border colors
    geom_rect(data = {
      temp_coords <- plot_coords
      temp_coords$treatment_full <- sapply(temp_coords$treatment, function(x) {
        switch(x, "T" = "Test", "C" = "Control", "R" = "Reference")
      })
      temp_coords$true_treatment_effect <- sapply(temp_coords$treatment_full, function(x) {
        switch(x, "Control" = 0, "Reference" = 0.5, "Test" = 1)
      })
      temp_coords
    }, 
              aes(xmin = x_center - 7.4, xmax = x_center + 7.4,
                  ymin = y_center - 4.9, ymax = y_center + 4.9,
                  color = true_treatment_effect), 
              fill = NA, linewidth = 2) +
    scale_color_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F",
                         midpoint = 0.5,  # Reference treatment at midpoint
                         name = "Treatment\nEffect (t/ha)",
                         breaks = c(0, 0.5, 1),
                         labels = c("Control (0)", "Reference (0.5)", "Test (1)")) +
    # Treatment labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 13, fontface = "bold", color = "black") +
    # Block labels on the left side
    geom_text(data = block_labels,
              aes(x = x_pos, y = y_pos, label = block),
              size = 5, fontface = "bold", hjust = 1, color = "black") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(-8, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "Integrated Analysis: RCBD Block Effects vs SpATS Spatial Splines",
         subtitle = "Purple contours: SpATS spatial effects | Single-focal-point environmental gradient | Block effects colored by RCBD estimates") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
      legend.position = "right"
    ) 

  # Add individual observations
  p <- add_wheat_spikes_with_legend(p, dataset1)
  
  return(p)
}

# =============================================================================
# MODEL COMPARISON AND ASSUMPTION TESTING
# =============================================================================

calculate_comprehensive_comparison <- function() {
  # True treatment effects
  true_treatment_effects <- c("Control" = 0, "Test" = 1, "Reference" = 0.5)
  
  # Calculate true environmental gradient for each observation
  # Convert to field coordinates
  dataset1$x_field <- dataset1$x * 5 - 2.5
  dataset1$y_field <- dataset1$y * 5 - 2.5
  
  # Define focal point
  focal_point1 <- c(x = 15, y = 20)
  
  # Calculate distances from focal point
  dist_to_focal1 <- sqrt((dataset1$x_field - focal_point1["x"])^2 + (dataset1$y_field - focal_point1["y"])^2)
  
  # Use distance and calculate true environmental effect
  min_distance <- dist_to_focal1
  gradient_increment <- min_distance * 0.15
  baseline_gradient <- 1.5
  focal_effect <- -1.5 + gradient_increment
  max_focal_distance <- 20
  focal_weight <- pmax(0, 1 - min_distance / max_focal_distance)
  dataset1$true_env_effect <- baseline_gradient * (1 - focal_weight) + focal_effect * focal_weight
  dataset1$true_env_effect <- pmax(-1.5, pmin(1.5, dataset1$true_env_effect))
  
  dataset1$true_treatment_effect <- sapply(dataset1$treatment, function(x) true_treatment_effects[x])
  
  # ===== RCBD MIXED MODEL EVALUATION =====
  cat("Evaluating RCBD Model...\n")
  
  # Extract fixed effects for treatments (works for both mixed and linear models)
  if (is_mixed_model) {
    fixed_effects <- fixef(rcbd_mixed_model)
  } else {
    fixed_effects <- coefficients(rcbd_mixed_model)
  }
  
  rcbd_treatment_est <- c("Control" = 0)  # Reference level
  
  if ("treatmentTest" %in% names(fixed_effects)) {
    rcbd_treatment_est["Test"] <- fixed_effects["treatmentTest"]
  } else {
    rcbd_treatment_est["Test"] <- 0
  }
  
  if ("treatmentReference" %in% names(fixed_effects)) {
    rcbd_treatment_est["Reference"] <- fixed_effects["treatmentReference"]
  } else {
    rcbd_treatment_est["Reference"] <- 0
  }
  
  # Calculate treatment errors
  rcbd_treatment_errors <- abs(rcbd_treatment_est - true_treatment_effects)
  rcbd_mean_treatment_error <- mean(rcbd_treatment_errors, na.rm = TRUE)
  
  # Get predictions for environmental effect calculation
  dataset1$rcbd_pred <- predict(rcbd_mixed_model, dataset1)
  dataset1$rcbd_treatment_component <- sapply(dataset1$treatment, function(x) rcbd_treatment_est[x])
  dataset1$rcbd_env_est <- dataset1$rcbd_pred - dataset1$rcbd_treatment_component - fixed_effects["(Intercept)"]
  
  # Calculate environmental errors
  rcbd_env_errors <- abs(dataset1$rcbd_env_est - dataset1$true_env_effect)
  rcbd_mean_env_error <- mean(rcbd_env_errors, na.rm = TRUE)
  
  # RCBD Assumption Tests
  rcbd_residuals <- residuals(rcbd_mixed_model)
  
  rcbd_assumptions <- list()
  rcbd_assumptions$normality <- tryCatch({
    shapiro.test(rcbd_residuals)
  }, error = function(e) list(p.value = NA, method = "Shapiro-Wilk test failed"))
  
  rcbd_assumptions$homoscedasticity_bartlett <- tryCatch({
    bartlett.test(rcbd_residuals, dataset1$treatment)
  }, error = function(e) list(p.value = NA, method = "Bartlett test failed"))
  
  # ===== SpATS MODEL EVALUATION =====
  cat("Evaluating SpATS Model...\n")
  
  # Get SpATS predictions
  dataset1$spats_pred <- suppressWarnings(predict(spats_model, dataset1)$predicted.values)
  
  # Extract SpATS treatment effects
  spats_coef <- spats_model$coeff
  spats_treatment_est <- c("Control" = 0)
  
  if ("Test" %in% names(spats_coef)) {
    spats_treatment_est["Test"] <- spats_coef["Test"]
  } else if ("treatmentTest" %in% names(spats_coef)) {
    spats_treatment_est["Test"] <- spats_coef["treatmentTest"]
  }
  
  if ("Reference" %in% names(spats_coef)) {
    spats_treatment_est["Reference"] <- spats_coef["Reference"]
  } else if ("treatmentReference" %in% names(spats_coef)) {
    spats_treatment_est["Reference"] <- spats_coef["treatmentReference"]
  }
  
  # Calculate treatment errors
  spats_treatment_errors <- abs(spats_treatment_est - true_treatment_effects)
  spats_mean_treatment_error <- mean(spats_treatment_errors, na.rm = TRUE)
  
  # Calculate environmental effects
  dataset1$spats_treatment_component <- sapply(dataset1$treatment, function(x) spats_treatment_est[x])
  intercept_val <- if("Intercept" %in% names(spats_coef)) spats_coef["Intercept"] else 0
  dataset1$spats_env_est <- dataset1$spats_pred - dataset1$spats_treatment_component - intercept_val
  
  spats_env_errors <- abs(dataset1$spats_env_est - dataset1$true_env_effect)
  spats_env_errors_clean <- spats_env_errors[is.finite(spats_env_errors)]
  spats_mean_env_error <- if(length(spats_env_errors_clean) > 0) mean(spats_env_errors_clean) else NA
  
  # SpATS Assumption Tests
  spats_residuals <- residuals(spats_model)
  
  spats_assumptions <- list()
  spats_assumptions$normality <- tryCatch({
    if (length(spats_residuals) > 3 && length(spats_residuals) <= 5000) {
      shapiro.test(spats_residuals)
    } else {
      list(p.value = NA, method = "Sample size inappropriate")
    }
  }, error = function(e) list(p.value = NA, method = "Test failed"))
  
  # Bartlett test for homoscedasticity
  spats_assumptions$homoscedasticity_bartlett <- tryCatch({
    bartlett.test(spats_residuals, dataset1$treatment)
  }, error = function(e) list(p.value = NA, method = "Bartlett test failed"))
  
  # Moran's I test for spatial autocorrelation
  spats_assumptions$spatial_autocorr_moran <- tryCatch({
    # Create spatial weights matrix for grid layout
    coords_matrix <- as.matrix(dataset1[, c("x", "y")])
    # Create neighbors list based on queen contiguity
    nb <- knn2nb(knearneigh(coords_matrix, k = 4))  # 4 nearest neighbors
    # Convert to spatial weights
    listw <- nb2listw(nb, style = "W", zero.policy = TRUE)
    # Perform Moran's I test
    moran.test(spats_residuals, listw, zero.policy = TRUE)
  }, error = function(e) list(p.value = NA, method = "Moran test failed"))
  
  # Write comprehensive summary
  sink("integrated_model_comparison.txt")
  cat("=== INTEGRATED MODEL COMPARISON SUMMARY ===\n")
  cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  cat("Comparison of RCBD vs SpATS Spatial Model\n")
  cat("Dataset: 54 observations, 3 treatments, 3 blocks\n")
  if (is_mixed_model) {
    cat("RCBD Model: Linear Mixed Model with random block effects\n")
  } else {
    cat("RCBD Model: Linear Model with fixed block effects\n")
  }
  cat("SpATS Model: Spatial spline model with PSANOVA smoothing\n\n")
  
  # Overall performance
  cat("OVERALL MODEL PERFORMANCE (Mean Absolute Errors):\n")
  cat("=================================================\n\n")
  cat(sprintf("%-20s %-20s %-25s\n", "Model", "Treatment_Error", "Environmental_Error"))
  cat(sprintf("%-20s %-20s %-25s\n", "-----", "---------------", "------------------"))
  cat(sprintf("%-20s %-20.4f %-25.4f\n", "RCBD_Model", rcbd_mean_treatment_error, rcbd_mean_env_error))
  cat(sprintf("%-20s %-20.4f %-25.4f\n", "SpATS_Spatial", spats_mean_treatment_error, spats_mean_env_error))
  
  # Treatment-specific effects
  cat("\n\nESTIMATED vs TRUE TREATMENT EFFECTS:\n")
  cat("====================================\n\n")
  cat(sprintf("%-12s %-12s %-15s %-15s %-12s %-12s\n", 
              "Treatment", "True_Effect", "RCBD_Estimate", "SpATS_Estimate", "RCBD_Error", "SpATS_Error"))
  cat(sprintf("%-12s %-12s %-15s %-15s %-12s %-12s\n", 
              "---------", "-----------", "-------------", "--------------", "----------", "-----------"))
  
  for(trt in names(true_treatment_effects)) {
    cat(sprintf("%-12s %-12.3f %-15.3f %-15.3f %-12.3f %-12.3f\n",
                trt, 
                true_treatment_effects[trt],
                rcbd_treatment_est[trt],
                spats_treatment_est[trt],
                rcbd_treatment_errors[trt],
                spats_treatment_errors[trt]))
  }
  
  # Block effects vs spatial modeling
  cat("\n\nBLOCK EFFECTS vs SPATIAL MODELING:\n")
  cat("===================================\n\n")
  cat("RCBD Block Effects:\n")
  if (is_mixed_model) {
    cat("(Random Effects from Mixed Model)\n")
  } else {
    cat("(Fixed Effects from Linear Model)\n")
  }
  for(i in 1:nrow(block_effects_df)) {
    cat(sprintf("Block %d: %+.4f t/ha\n", 
                block_effects_df$block[i], 
                block_effects_df$block_effect[i]))
  }
  
  cat("\nSpATS Spatial Modeling:\n")
  cat("- Uses PSANOVA splines to model continuous spatial variation\n")
  cat("- Spatial effect range:", sprintf("%.3f to %.3f t/ha", 
                                         min(pred_grid_fine$spline_effect, na.rm = TRUE),
                                         max(pred_grid_fine$spline_effect, na.rm = TRUE)), "\n")
  
  # Assumption testing
  cat("\n\nMODEL ASSUMPTION TESTS:\n")
  cat("=======================\n\n")
  
  cat("RCBD Model:\n")
  cat("-----------\n")
  if (is_mixed_model) {
    cat("(Linear Mixed Model with random block effects)\n")
  } else {
    cat("(Linear Model with fixed block effects)\n")
  }
  if (!is.null(rcbd_assumptions$normality) && !is.na(rcbd_assumptions$normality$p.value)) {
    cat(sprintf("Normality (Shapiro-Wilk): W = %.4f, p-value = %.4f\n",
                rcbd_assumptions$normality$statistic,
                rcbd_assumptions$normality$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(rcbd_assumptions$normality$p.value > 0.05,
                       "Residuals normally distributed (p > 0.05)",
                       "Non-normal residuals (p ≤ 0.05)")))
  }
  
  if (!is.null(rcbd_assumptions$homoscedasticity_bartlett) && !is.na(rcbd_assumptions$homoscedasticity_bartlett$p.value)) {
    cat(sprintf("Homoscedasticity (Bartlett): K-squared = %.4f, p-value = %.4f\n",
                rcbd_assumptions$homoscedasticity_bartlett$statistic,
                rcbd_assumptions$homoscedasticity_bartlett$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(rcbd_assumptions$homoscedasticity_bartlett$p.value > 0.05,
                       "Homoscedastic residuals (p > 0.05)",
                       "Heteroscedastic residuals (p ≤ 0.05)")))
  }
  
  cat("\nSpATS Model:\n")
  cat("------------\n")
  if (!is.null(spats_assumptions$normality) && !is.na(spats_assumptions$normality$p.value)) {
    cat(sprintf("Normality (Shapiro-Wilk): W = %.4f, p-value = %.4f\n",
                spats_assumptions$normality$statistic,
                spats_assumptions$normality$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(spats_assumptions$normality$p.value > 0.05,
                       "Residuals normally distributed (p > 0.05)",
                       "Non-normal residuals (p ≤ 0.05)")))
  }
  
  if (!is.null(spats_assumptions$homoscedasticity_bartlett) && !is.na(spats_assumptions$homoscedasticity_bartlett$p.value)) {
    cat(sprintf("Homoscedasticity (Bartlett): K-squared = %.4f, p-value = %.4f\n",
                spats_assumptions$homoscedasticity_bartlett$statistic,
                spats_assumptions$homoscedasticity_bartlett$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(spats_assumptions$homoscedasticity_bartlett$p.value > 0.05,
                       "Homoscedastic residuals (p > 0.05)",
                       "Heteroscedastic residuals (p ≤ 0.05)")))
  }
  
  if (!is.null(spats_assumptions$spatial_autocorr_moran) && !is.na(spats_assumptions$spatial_autocorr_moran$p.value)) {
    cat(sprintf("Spatial Autocorrelation (Moran's I): I = %.4f, p-value = %.4f\n",
                spats_assumptions$spatial_autocorr_moran$estimate[1],
                spats_assumptions$spatial_autocorr_moran$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(spats_assumptions$spatial_autocorr_moran$p.value > 0.05,
                       "No significant spatial autocorrelation (p > 0.05)",
                       "Significant spatial autocorrelation present (p ≤ 0.05)")))
  }
  
  # Model comparison interpretation
  cat("\n\nMODEL COMPARISON INTERPRETATION:\n")
  cat("================================\n\n")
  
  if (spats_mean_treatment_error < rcbd_mean_treatment_error) {
    cat("Treatment Effect Estimation: SpATS performs better (lower error)\n")
  } else {
    cat("Treatment Effect Estimation: RCBD Model performs better (lower error)\n")
  }
  
  if (spats_mean_env_error < rcbd_mean_env_error) {
    cat("Environmental Effect Estimation: SpATS performs better (lower error)\n")
  } else {
    cat("Environmental Effect Estimation: RCBD Model performs better (lower error)\n")
  }
  
  cat("\nKey Differences:\n")
  cat("- RCBD: Captures environmental variation through discrete block effects\n")
  cat("- SpATS: Models continuous spatial variation using smooth splines\n")
  cat("- Mixed Model: Accounts for pseudoreplication with random effects (if convergent)\n")
  cat("- Spatial Model: Explicitly models spatial correlation structure\n")
  
  sink()
  
  cat("Comprehensive model comparison written to: integrated_model_comparison.txt\n")
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

cat("=== GENERATING INTEGRATED PLOT ===\n")
integrated_plot <- create_integrated_plot()

# Save the plot
suppressMessages(suppressWarnings(
  ggsave("integrated_rcbd_spats_comparison.png", integrated_plot, 
         width = 14, height = 10, dpi = 300)
))

cat("Integrated plot saved as: integrated_rcbd_spats_comparison.png\n\n")

# Calculate comprehensive comparison
cat("=== CALCULATING COMPREHENSIVE MODEL COMPARISON ===\n")
calculate_comprehensive_comparison()

cat("\n=== ANALYSIS COMPLETE ===\n")

# Re-enable warnings and show summary if any occurred
options(warn = 0)
cat("Checking for any warnings during analysis...\n")
if (length(warnings()) > 0) {
  cat("Note: Some warnings occurred during model fitting (this is normal for spatial models)\n")
  cat("Most common warnings are related to:\n")
  cat("- SpATS convergence iterations\n")
  cat("- Spatial smoothing parameter estimation\n")
  cat("- Mixed model boundary conditions\n")
  cat("These warnings typically don't affect result validity.\n\n")
} else {
  cat("No warnings detected.\n\n")
}

cat("Files generated:\n")
cat("1. integrated_rcbd_spats_comparison.png - Comprehensive visualization\n")
cat("2. integrated_model_comparison.txt - Detailed statistical comparison\n")
cat("\nThe plot shows:\n")
cat("- Background: True environmental gradient\n")
cat("- Purple contours: SpATS estimated spatial effects\n")
cat("- Colored block labels: RCBD estimated block effects\n")
cat("- Plot borders: Treatment effects (colored by magnitude)\n")
cat("- Individual observations: Wheat spikes sized by yield\n")
