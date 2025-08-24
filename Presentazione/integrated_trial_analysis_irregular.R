# Integrated Trial Analysis: RCBD vs SpATS Comparison (Irregular Environmental Gradient)
# This script generates trial data with irregular environmental gradient matching the presentation slide
# showing both block effects (RCBD) and spatial spline effects (SpATS)

# Set library path to user directory
#.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# Load required packages
# Set library path to user directory
#.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# Function to check and install required packages
check_and_install_packages <- function() {
  # List of required packages with installation order (dependencies first)
  required_packages <- c(
    "ggplot2",      # For plotting
    "dplyr",        # For data manipulation
    "gridExtra",    # For arranging plots
    "metR",         # For geom_text_contour
    "SpATS",        # For spatial modeling
    "gstat",        # For variogram modeling
    "sp",           # For spatial data structures
    "lme4",         # For linear mixed models
    "car",          # For Levene's test
    "lmtest",       # For Durbin-Watson test
    "spdep",        # For Moran's I test
    "scales"        # For rescale function (used in visualization)
  )
  
  cat("Checking required packages...\n")
  
  # Function to clean lock directories
  clean_lock_dirs <- function() {
    lib_path <- .libPaths()[1]
    lock_dirs <- list.dirs(lib_path, full.names = TRUE, recursive = FALSE)
    lock_dirs <- lock_dirs[grepl("00LOCK", basename(lock_dirs))]
    
    if (length(lock_dirs) > 0) {
      cat("Cleaning lock directories...\n")
      for (lock_dir in lock_dirs) {
        try(unlink(lock_dir, recursive = TRUE, force = TRUE), silent = TRUE)
        cat("Removed:", basename(lock_dir), "\n")
      }
    }
  }
  
  # Clean any existing lock directories
  clean_lock_dirs()
  
  # Check which packages are not installed
  missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
  
  if (length(missing_packages) > 0) {
    cat("Missing packages detected:", paste(missing_packages, collapse = ", "), "\n")
    cat("Installing missing packages...\n")
    
    # Install RcppEigen first if needed (common dependency)
    if ("lme4" %in% missing_packages || "car" %in% missing_packages) {
      if (!"RcppEigen" %in% installed.packages()[,"Package"]) {
        cat("Installing RcppEigen (required dependency)...\n")
        tryCatch({
          install.packages("RcppEigen", dependencies = TRUE)
          cat("RcppEigen installed successfully.\n")
        }, error = function(e) {
          cat("Error installing RcppEigen:", e$message, "\n")
        })
      }
    }
    
    # Install missing packages one by one
    for (pkg in missing_packages) {
      if (!pkg %in% installed.packages()[,"Package"]) {
        cat("Installing", pkg, "...\n")
        tryCatch({
          # Clean lock dirs before each installation
          clean_lock_dirs()
          
          install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org")
          cat(pkg, "installed successfully.\n")
        }, error = function(e) {
          cat("Error installing", pkg, ":", e$message, "\n")
          
          # Try alternative installation method
          cat("Trying alternative installation for", pkg, "...\n")
          tryCatch({
            clean_lock_dirs()
            install.packages(pkg, dependencies = FALSE, repos = "https://cloud.r-project.org")
            cat(pkg, "installed successfully (without dependencies).\n")
          }, error = function(e2) {
            cat("Alternative installation also failed for", pkg, "\n")
          })
        })
      }
    }
  } else {
    cat("All required packages are already installed.\n")
  }
  
  # Verify all packages can be loaded
  cat("\nVerifying package loading...\n")
  failed_packages <- c()
  
  for (pkg in required_packages) {
    tryCatch({
      library(pkg, character.only = TRUE, quietly = TRUE)
      cat("✓", pkg, "loaded successfully\n")
    }, error = function(e) {
      cat("✗", pkg, "failed to load:", e$message, "\n")
      failed_packages <<- c(failed_packages, pkg)
    })
  }
  
  if (length(failed_packages) > 0) {
    cat("\nWarning: The following packages could not be loaded:\n")
    cat(paste(failed_packages, collapse = ", "), "\n")
    
    # Try to install failed packages one more time
    cat("Attempting to reinstall failed packages...\n")
    for (pkg in failed_packages) {
      tryCatch({
        clean_lock_dirs()
        remove.packages(pkg, lib = .libPaths()[1])
      }, error = function(e) {})
      
      tryCatch({
        install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org")
        library(pkg, character.only = TRUE, quietly = TRUE)
        cat("✓", pkg, "reinstalled and loaded successfully\n")
        failed_packages <<- failed_packages[failed_packages != pkg]
      }, error = function(e) {
        cat("✗", pkg, "still failed after reinstall\n")
      })
    }
    
    if (length(failed_packages) > 0) {
      cat("\nSome packages still cannot be loaded. Continuing with available packages...\n")
      return(FALSE)
    }
  }
  
  cat("\nAll packages loaded successfully!\n\n")
  return(TRUE)
}

# Run the package check and installation
cat("Starting package installation and verification...\n")
if (!check_and_install_packages()) {
  cat("Some packages could not be installed, but continuing with available packages...\n")
  cat("You may need to install missing packages manually later.\n\n")
}

# Load required packages (this section replaces your original library() calls)
# library(ggplot2)
# library(dplyr)
# library(gridExtra)
# library(metR)      # For geom_text_contour
# library(SpATS)     # For spatial modeling
# library(gstat)     # For variogram modeling
# library(sp)        # For spatial data structures
# library(lme4)      # For linear mixed models
# library(car)       # For Levene's test
# library(lmtest)    # For Durbin-Watson test
# library(spdep)     # For Moran's I test

# Set seed for reproducibility
set.seed(123)

# Configure emoji font support for better rendering
if (Sys.info()["sysname"] == "Linux") {
  # Try to ensure emoji fonts are available
  cat("Configuring emoji font support for Linux...\n")
  
  # Check if system has emoji fonts installed
  emoji_fonts <- c("Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", "Symbola")
  
  # Set graphics device options for better Unicode support
  options(device = function(...) {
    if (requireNamespace("ragg", quietly = TRUE)) {
      ragg::agg_png(..., res = 300)
    } else {
      png(..., res = 300, type = "cairo")
    }
  })
  
  cat("Emoji font configuration completed.\n")
}

# Suppress common warnings for cleaner output
# options(warn = -1)  # Temporarily suppress warnings

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

# Spatial gradient parameters - irregular pattern matching slide
spatial_gradient_range <- c(-1.5, 1.5)  # tons/ha
base_yield <- 12  # tons/ha (baseline yield)

# Create spatial coordinates (regular grid)
x_coords <- rep(1:n_cols, n_rows)
y_coords <- rep(1:n_rows, each = n_cols)
coordinates <- data.frame(x = x_coords, y = y_coords)

# Create irregular spatial gradient matching the presentation slide pattern
# Convert plot coordinates to field coordinates (multiply by 5 and adjust)
x_field <- coordinates$x * 5 - 2.5  # Convert to field scale
y_field <- coordinates$y * 5 - 2.5  # Convert to field scale

# Normalize coordinates to match TikZ coordinate system (0-6 for x, 0-4.5 for y approximately)
x_tikz <- (x_field + 2.5) * 6 / 45  # Scale to 0-6 range
y_tikz <- (y_field + 2.5) * 4.5 / 30  # Scale to 0-4.5 range

# Define irregular environmental zones based on TikZ patterns from slide
# Initialize with medium gradient (0) as default
spatial_gradient <- rep(0, nrow(coordinates))

# Define the irregular zones using polygon membership functions
# LOW environmental gradient (-1.5) - irregular diagonal patches (bottom area)
is_in_low_zone <- function(x, y) {
  # Approximate the irregular polygon from TikZ: 
  # \fill[envlow] ({0+\xoffset},{0+\yoffset}) -- ({2+\xoffset},{1+\yoffset}) -- ({3+\xoffset},{0.5+\yoffset}) -- ({4+\xoffset},{1.5+\yoffset}) -- ({6+\xoffset},{1+\yoffset}) -- ({6+\xoffset},{-0.5+\yoffset}) -- ({0+\xoffset},{-0.5+\yoffset}) -- cycle;
  
  # Check if point is within the irregular low zone (simplified polygon approximation)
  # Lower boundary: y >= -0.5
  # Upper boundary: complex curve approximately following the zigzag pattern
  lower_bound <- -0.5
  
  # Approximate upper boundary with piecewise linear function
  if (x <= 2) {
    upper_bound <- 0 + 0.5 * x  # Linear increase from 0 to 1
  } else if (x <= 3) {
    upper_bound <- 1 - 0.5 * (x - 2)  # Decrease from 1 to 0.5
  } else if (x <= 4) {
    upper_bound <- 0.5 + 1 * (x - 3)  # Increase from 0.5 to 1.5
  } else {
    upper_bound <- 1.5 - 0.5 * (x - 4)  # Decrease from 1.5 to 1
  }
  
  return(y >= lower_bound & y <= upper_bound)
}

# HIGH environmental gradient (+1.5) - irregular top patches
is_in_high_zone <- function(x, y) {
  # Approximate the irregular polygon from TikZ:
  # \fill[envhigh] ({-0.5+\xoffset},{2.8+\yoffset}) -- ({2+\xoffset},{3.5+\yoffset}) -- ({4+\xoffset},{3.2+\yoffset}) -- ({6.5+\xoffset},{4.5+\yoffset}) -- ({6.5+\xoffset},{2.5+\yoffset}) -- ({5+\xoffset},{2.8+\yoffset}) -- ({3.5+\xoffset},{2.2+\yoffset}) -- ({1.5+\xoffset},{1.8+\yoffset}) -- ({-0.5+\xoffset},{1.2+\yoffset}) -- cycle;
  
  # Lower boundary: complex curve
  if (x <= 1.5) {
    lower_bound <- 1.2 + 0.4 * x  # Increase from 1.2 to 1.8
  } else if (x <= 3.5) {
    lower_bound <- 1.8 + 0.2 * (x - 1.5)  # Increase from 1.8 to 2.2
  } else if (x <= 5) {
    lower_bound <- 2.2 + 0.4 * (x - 3.5)  # Increase from 2.2 to 2.8
  } else {
    lower_bound <- 2.8 - 0.3 * (x - 5)  # Decrease from 2.8 to 2.5
  }
  
  # Upper boundary: complex curve
  if (x <= 2) {
    upper_bound <- 2.8 + 0.35 * x  # Increase from 2.8 to 3.5
  } else if (x <= 4) {
    upper_bound <- 3.5 - 0.15 * (x - 2)  # Decrease from 3.5 to 3.2
  } else {
    upper_bound <- 3.2 + 0.65 * (x - 4)  # Increase from 3.2 to 4.5
  }
  
  return(y >= lower_bound & y <= upper_bound)
}

# Apply the irregular zone classification
for (i in 1:nrow(coordinates)) {
  x <- x_tikz[i]
  y <- y_tikz[i]
  
  if (is_in_low_zone(x, y)) {
    spatial_gradient[i] <- -1.5  # Low environmental gradient
  } else if (is_in_high_zone(x, y)) {
    spatial_gradient[i] <- 1.5   # High environmental gradient
  }
  # else remains 0 (medium environmental gradient)
}

# Assign treatments to plots (Proper RCBD design)
# Create a proper randomized complete block design
set.seed(456)  # Different seed for treatment assignment to avoid confounding

# First, determine blocks based on rows
block_assignment <- function(y) {
  if (y <= 2) {
    return(1)
  } else if (y <= 4) {
    return(2)
  } else {
    return(3)
  }
}

# Apply block assignment
blocks <- sapply(coordinates$y, block_assignment)

# Create proper RCBD treatment assignment
# Each block should have equal representation of each treatment
treatments <- rep(NA, nrow(coordinates))

for (block_num in 1:3) {
  block_indices <- which(blocks == block_num)
  n_plots_per_block <- length(block_indices)
  
  # Create balanced treatment assignment within block
  n_per_treatment <- n_plots_per_block / 3
  block_treatments <- rep(c("Control", "Test", "Reference"), each = n_per_treatment)
  
  # Randomize the order within block
  block_treatments <- sample(block_treatments)
  
  # Assign to the block
  treatments[block_indices] <- block_treatments
}

# Calculate treatment effects
treatment_effect_values <- treatment_effects[treatments]

# Create heteroscedastic pattern for RCBD to fail assumption tests
# Add position-dependent variance multiplier
x_norm <- (coordinates$x - 1) / (9 - 1)  # Normalize x coordinates to 0-1
y_norm <- (coordinates$y - 1) / (6 - 1)  # Normalize y coordinates to 0-1

# Random error generation
base_error_sd <- 0.2
random_error <- rnorm(nrow(coordinates), mean = 0, sd = base_error_sd)

# Calculate final yield values
yield_values <- base_yield + spatial_gradient + treatment_effect_values + random_error

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
cat("Yield range:", round(range(dataset1$yield), 2), "tons/ha\n")
cat("Environmental gradient values:", unique(dataset1$spatial_gradient), "tons/ha\n\n")

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
  
  # Add wheat spike annotations with corn emoji
  for(i in 1:nrow(individual_data)) {
    size_val <- switch(individual_data$yield_category[i],
                       "Low" = 3, "Medium" = 5, "High" = 7)
    plot_obj <- plot_obj + 
      annotate("text", x = individual_data$x_scaled[i], y = individual_data$y_scaled[i], 
               label = "🌽", size = size_val, hjust = 0.5, vjust = 0.5,
               family = "Noto Color Emoji")  # Explicitly specify emoji font
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
          shape = c("🌽", "🌽", "🌽")[1:n_levels],  # Use corn emoji in legend
          color = c("goldenrod4", "goldenrod3", "goldenrod1")[1:n_levels],  # Use corn-like colors
          size = c(4, 6, 8)[1:n_levels],  # Use only the needed sizes
          family = "Noto Color Emoji"  # Specify emoji font for legend
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
  # Environmental gradient background with irregular pattern matching slide
  x_seq <- seq(0, 45, length.out = 100)
  y_seq <- seq(0, 30, length.out = 100)
  grid_data <- expand.grid(x = x_seq, y = y_seq)
  
  # Convert to TikZ coordinate system for zone classification
  x_tikz_grid <- (grid_data$x + 2.5) * 6 / 45
  y_tikz_grid <- (grid_data$y + 2.5) * 4.5 / 30
  
  # Initialize with medium gradient (0) as default
  grid_data$env_gradient <- 0
  
  # Apply the irregular zone classification to grid
  for (i in 1:nrow(grid_data)) {
    x <- x_tikz_grid[i]
    y <- y_tikz_grid[i]
    
    if (is_in_low_zone(x, y)) {
      grid_data$env_gradient[i] <- -1.5  # Low environmental gradient
    } else if (is_in_high_zone(x, y)) {
      grid_data$env_gradient[i] <- 1.5   # High environmental gradient
    }
    # else remains 0 (medium environmental gradient)
  }
  
  # Debug: Print gradient range for verification
  cat("Environmental gradient range:", range(grid_data$env_gradient, na.rm = TRUE), "\n")
  cat("Gradient values summary:\n")
  print(table(grid_data$env_gradient))
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "Environmental\nSpatial Effect\n(t/ha)",
                        breaks = c(-1.5, 0, 1.5), labels = c("Low (-1.5)", "Medium (0)", "High (+1.5)")) +
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
    # Spline effect contours from SpATS model (only for finite values)
    geom_contour(data = pred_grid_fine[is.finite(pred_grid_fine$spline_effect), ], 
                aes(x = x_scaled, y = y_scaled, z = spline_effect),
                color = "purple", linewidth = 1.5, alpha = 0.8,
                breaks = seq(10, 14, 0.2)) +
    # Spline effect contour labels (only for finite values) - using geom_text_contour for better control
    geom_text_contour(data = pred_grid_fine[is.finite(pred_grid_fine$spline_effect), ], 
                     aes(x = x_scaled, y = y_scaled, z = spline_effect),
                     color = "purple4", size = 4, fontface = "bold",
                     breaks = seq(10, 14, 0.2), 
                     skip = 1,
                     label.formatter = function(x) sprintf("%.1f", x)) +
    # Add invisible line for contour legend
    geom_line(data = data.frame(x = c(-100, -101), y = c(-100, -101)), 
              aes(x = x, y = y, linetype = "SpATS Spatial Effects"), 
              color = "purple", linewidth = 1.5, alpha = 0) +
    scale_linetype_manual(name = "Model Contours", 
                         values = c("SpATS Spatial Effects" = "solid"),
                         guide = guide_legend(
                           override.aes = list(alpha = 1, color = "purple", linewidth = 1.5),
                           title.position = "top",
                           title.hjust = 0.5
                         )) +
    # Treatment labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 13, fontface = "bold", color = "black") +
    # Block labels on the left side
    geom_text(data = block_labels,
              aes(x = x_pos, y = y_pos, label = block),
              size = 5, fontface = "bold", hjust = 1, color = "black") +
    # Block effect values below each block label in dark violet
        geom_text(data = {
          # Calculate grand mean and add it to block effects
          if (is_mixed_model) {
            grand_mean <- fixef(rcbd_mixed_model)["(Intercept)"]
          } else {
            grand_mean <- coefficients(rcbd_mixed_model)["(Intercept)"]
          }
          
          # Add grand mean to block effects to get absolute block means
          block_labels_with_means <- block_labels
          block_labels_with_means$absolute_block_mean <- grand_mean + block_labels_with_means$block_effect
          block_labels_with_means
        },
                  aes(x = x_pos, y = y_pos - 2, label = sprintf("%.1f t/ha", absolute_block_mean)),
                  size = 4, fontface = "bold", hjust = 1, color = "darkviolet") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(-8, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "Trial Analysis: RCBD vs SpATS Comparison",
         subtitle = "Purple contours with values: SpATS spatial effects (t/ha) | Background: Environmental zones | Plot borders: Treatment effects",
         caption = "🌽 = Individual observations (size indicates yield level)\nTreatments: T=Test, C=Control, R=Reference") +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 11, color = "gray40"),
      plot.caption = element_text(size = 10, color = "gray60", hjust = 0),
      axis.title = element_text(size = 12),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid = element_line(color = "gray90", linewidth = 0.5)
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
  true_treatment_effects <- c("Control" = 0, "Reference" = 0.5, "Test" = 1)
  
  # Define consistent treatment order
  treatment_names <- c("Control", "Reference", "Test")
  
  # The environmental gradient is now discrete (irregular zones)
  dataset1$true_env_effect <- dataset1$spatial_gradient
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
  
  # Calculate treatment errors - ensure proper alignment
  # Order both vectors the same way
  rcbd_treatment_est_ordered <- rcbd_treatment_est[treatment_names]
  true_treatment_effects_ordered <- true_treatment_effects[treatment_names]
  
  rcbd_treatment_errors <- abs(rcbd_treatment_est_ordered - true_treatment_effects_ordered)
  names(rcbd_treatment_errors) <- treatment_names
  rcbd_mean_treatment_error <- mean(rcbd_treatment_errors, na.rm = TRUE)
  
  # Get predictions for environmental effect calculation
  dataset1$rcbd_pred <- predict(rcbd_mixed_model, dataset1)
  dataset1$rcbd_treatment_component <- sapply(dataset1$treatment, function(x) rcbd_treatment_est[x])
  dataset1$rcbd_env_est <- dataset1$rcbd_pred - dataset1$rcbd_treatment_component - fixed_effects["(Intercept)"]
  
  # Calculate environmental errors
  rcbd_env_errors <- abs(dataset1$rcbd_env_est - dataset1$true_env_effect)
  rcbd_mean_env_error <- mean(rcbd_env_errors, na.rm = TRUE)
  
  # Debug: Print treatment estimates and errors for verification
  cat("RCBD Treatment estimates:\n")
  print(rcbd_treatment_est)
  cat("True treatment effects:\n")
  print(true_treatment_effects)
  cat("RCBD Treatment errors:\n")
  print(rcbd_treatment_errors)
  
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
  
  # Calculate treatment errors - ensure proper alignment
  # Order both vectors the same way
  spats_treatment_est_ordered <- spats_treatment_est[treatment_names]
  true_treatment_effects_ordered <- true_treatment_effects[treatment_names]
  
  spats_treatment_errors <- abs(spats_treatment_est_ordered - true_treatment_effects_ordered)
  names(spats_treatment_errors) <- treatment_names
  spats_mean_treatment_error <- mean(spats_treatment_errors, na.rm = TRUE)
  
  # Debug: Print treatment estimates and errors for verification
  cat("SpATS Treatment estimates:\n")
  print(spats_treatment_est)
  cat("SpATS Treatment errors:\n")
  print(spats_treatment_errors)
  
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
      list(p.value = NA, method = "Shapiro-Wilk not applicable (sample size)")
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
  sink("integrated_model_comparison_irregular.txt")
  cat("=== INTEGRATED MODEL COMPARISON SUMMARY (IRREGULAR PATTERN) ===\n")
  cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  cat("Comparison of RCBD vs SpATS Spatial Model\n")
  cat("Dataset: 54 observations, 3 treatments, 3 blocks\n")
  cat("Environmental Pattern: Irregular zones matching presentation slide\n")
  cat("- Low zone: -1.5 t/ha (irregular diagonal patches)\n")
  cat("- Medium zone: 0 t/ha (middle areas)\n")
  cat("- High zone: +1.5 t/ha (irregular top patches)\n\n")
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
  
  for(trt in treatment_names) {
    cat(sprintf("%-12s %-12.3f %-15.3f %-15.3f %-12.3f %-12.3f\n",
                trt, true_treatment_effects_ordered[trt], rcbd_treatment_est_ordered[trt], spats_treatment_est_ordered[trt],
                rcbd_treatment_errors[trt], spats_treatment_errors[trt]))
  }
  
  # Environmental pattern analysis
  cat("\n\nENVIRONMENTAL PATTERN ANALYSIS:\n")
  cat("===============================\n\n")
  cat("Irregular Environmental Zones:\n")
  zone_counts <- table(dataset1$spatial_gradient)
  cat("- Low zone (-1.5 t/ha):", zone_counts["-1.5"], "observations\n")
  cat("- Medium zone (0 t/ha):", zone_counts["0"], "observations\n") 
  cat("- High zone (+1.5 t/ha):", zone_counts["1.5"], "observations\n\n")
  
  # Block effects vs spatial modeling
  cat("BLOCK EFFECTS vs SPATIAL MODELING:\n")
  cat("===================================\n\n")
  cat("RCBD Block Effects:\n")
  if (is_mixed_model) {
    cat("(Random Effects from Mixed Model)\n")
  } else {
    cat("(Fixed Effects from Linear Model)\n")
  }
  for(i in 1:nrow(block_effects_df)) {
    cat(sprintf("Block %d: %+.4f t/ha\n", 
                block_effects_df$block[i], block_effects_df$block_effect[i]))
  }
  
  cat("\nSpATS Spatial Modeling:\n")
  cat("- Uses PSANOVA splines to model continuous spatial variation\n")
  # Calculate relative spatial effects (remove intercept to show spatial variation)
  intercept_val <- if("Intercept" %in% names(spats_coef)) spats_coef["Intercept"] else mean(pred_grid_fine$spline_effect, na.rm = TRUE)
  relative_spatial_effects <- pred_grid_fine$spline_effect - intercept_val
  cat("- Spatial effect range:", sprintf("%.3f to %.3f t/ha", 
                                         min(relative_spatial_effects, na.rm = TRUE),
                                         max(relative_spatial_effects, na.rm = TRUE)), "\n")
  
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
                rcbd_assumptions$normality$statistic, rcbd_assumptions$normality$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(rcbd_assumptions$normality$p.value > 0.05, "Normality assumption satisfied", "Normality assumption violated")))
  }
  
  if (!is.null(rcbd_assumptions$homoscedasticity_bartlett) && !is.na(rcbd_assumptions$homoscedasticity_bartlett$p.value)) {
    cat(sprintf("Homoscedasticity (Bartlett): K-squared = %.4f, p-value = %.4f\n",
                rcbd_assumptions$homoscedasticity_bartlett$statistic, rcbd_assumptions$homoscedasticity_bartlett$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(rcbd_assumptions$homoscedasticity_bartlett$p.value > 0.05, "Homoscedasticity assumption satisfied", "Homoscedasticity assumption violated")))
  }
  
  cat("\nSpATS Model:\n")
  cat("------------\n")
  if (!is.null(spats_assumptions$normality) && !is.na(spats_assumptions$normality$p.value)) {
    cat(sprintf("Normality (Shapiro-Wilk): W = %.4f, p-value = %.4f\n",
                spats_assumptions$normality$statistic, spats_assumptions$normality$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(spats_assumptions$normality$p.value > 0.05, "Normality assumption satisfied", "Normality assumption violated")))
  }
  
  if (!is.null(spats_assumptions$homoscedasticity_bartlett) && !is.na(spats_assumptions$homoscedasticity_bartlett$p.value)) {
    cat(sprintf("Homoscedasticity (Bartlett): K-squared = %.4f, p-value = %.4f\n",
                spats_assumptions$homoscedasticity_bartlett$statistic, spats_assumptions$homoscedasticity_bartlett$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(spats_assumptions$homoscedasticity_bartlett$p.value > 0.05, "Homoscedasticity assumption satisfied", "Homoscedasticity assumption violated")))
  }
  
  if (!is.null(spats_assumptions$spatial_autocorr_moran) && !is.na(spats_assumptions$spatial_autocorr_moran$p.value)) {
    # Extract the Moran's I statistic (first element if multiple)
    moran_i_value <- if(is.numeric(spats_assumptions$spatial_autocorr_moran$estimate)) {
      spats_assumptions$spatial_autocorr_moran$estimate[1]
    } else {
      spats_assumptions$spatial_autocorr_moran$estimate["Moran I statistic"]
    }
    cat(sprintf("Spatial Autocorrelation (Moran's I): I = %.4f, p-value = %.4f\n",
                moran_i_value, spats_assumptions$spatial_autocorr_moran$p.value))
    cat(sprintf("Interpretation: %s\n",
                ifelse(spats_assumptions$spatial_autocorr_moran$p.value > 0.05, "No significant spatial autocorrelation", "Significant spatial autocorrelation detected")))
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
  
  cat("\nKey Differences with Irregular Pattern:\n")
  cat("- RCBD: Attempts to capture environmental variation through discrete block effects\n")
  cat("- SpATS: Models continuous spatial variation using smooth splines\n")
  cat("- Irregular Pattern: Creates challenging conditions for both models\n")
  cat("- Zone Classification: True environmental effects are discrete (-1.5, 0, +1.5)\n")
  cat("- Spatial Continuity: SpATS assumes smooth transitions; irregular zones violate this\n")
  
  sink()
  
  cat("Comprehensive model comparison written to: integrated_model_comparison_irregular.txt\n")
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

cat("=== GENERATING INTEGRATED PLOT WITH IRREGULAR ENVIRONMENTAL PATTERN ===\n")
integrated_plot <- create_integrated_plot()

# Save the plot with proper emoji support
suppressMessages(suppressWarnings({
  # Set up proper font configuration for emoji rendering
  # Check if we're on Linux and try to use fonts that support emoji
  if (Sys.info()["sysname"] == "Linux") {
    # Try to install and use ragg package for better font support
    if (!requireNamespace("ragg", quietly = TRUE)) {
      cat("Installing ragg package for better emoji support...\n")
      install.packages("ragg", quiet = TRUE)
    }
    
    if (requireNamespace("ragg", quietly = TRUE)) {
      # Use ragg with explicit font configuration
      ragg::agg_png("integrated_rcbd_spats_comparison_irregular.png", 
                    width = 14, height = 10, units = "in", res = 300,
                    scaling = 1.0)
      print(integrated_plot)
      dev.off()
      cat("Plot saved with ragg device (better emoji support)\n")
    } else {
      # Fallback: Try cairo with UTF-8 encoding
      tryCatch({
        cairo_png("integrated_rcbd_spats_comparison_irregular.png", 
                  width = 14*300, height = 10*300, res = 300)
        print(integrated_plot)
        dev.off()
        cat("Plot saved with cairo device\n")
      }, error = function(e) {
        # Final fallback
        ggsave("integrated_rcbd_spats_comparison_irregular.png", integrated_plot, 
               width = 14, height = 10, dpi = 300, device = "png")
        cat("Plot saved with standard png device\n")
      })
    }
  } else {
    # For non-Linux systems, use standard approach
    ggsave("integrated_rcbd_spats_comparison_irregular.png", integrated_plot, 
           width = 14, height = 10, dpi = 300, device = "png")
    cat("Plot saved with standard device\n")
  }
}))

cat("Integrated plot saved as: integrated_rcbd_spats_comparison_irregular.png\n\n")

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
cat("1. integrated_rcbd_spats_comparison_irregular.png - Comprehensive visualization\n")
cat("2. integrated_model_comparison_irregular.txt - Detailed statistical comparison\n")
cat("\nThe plot shows:\n")
cat("- Background: Irregular environmental gradient matching presentation slide\n")
cat("- Environmental zones: Low (-1.5), Medium (0), High (+1.5) t/ha\n")
cat("- Purple contours: SpATS estimated spatial effects\n")
cat("- Colored block labels: RCBD estimated block effects\n")
cat("- Plot borders: Treatment effects (colored by magnitude)\n")
cat("- Individual observations: Wheat spikes sized by yield\n")
