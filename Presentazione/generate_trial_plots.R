# Generate Trial Design Plots for Presentation
# This script creates PNG files showing trial design with environmental gradients
# and model-specific effects overlaid in different colors

# Set library path to user directory
.libPaths(c("C:/Users/samuele.bumbaca/Documents/R/win-library/4.5", .libPaths()))

# Load required packages
library(ggplot2)
library(dplyr)
library(gridExtra)
library(metR)  # For geom_text_contour
library(png)   # For reading PNG images
library(grid)  # For rasterGrob
library(SpATS) # For spatial modeling
library(gstat) # For variogram modeling
library(sp)    # For spatial data structures

# Set up plotting parameters
plot_width <- 12
plot_height <- 8
dpi <- 300

# Create directory for images if it doesn't exist
if (!dir.exists("Presentazione/Imgs")) {
  dir.create("Presentazione/Imgs")
}

# Load the datasets directly
dataset1 <- read.csv("dataset1_detailed.csv")
dataset2 <- read.csv("dataset2_plots.csv")

# Convert dataset2 coordinates to plot coordinates (scale from grid to field coordinates)
# Original grid: 9 cols x 6 rows -> Field: 45m x 30m
dataset2$x_plot <- dataset2$x_center * 5   # Scale x: 9 grid units -> 45m
dataset2$y_plot <- dataset2$y_center * 5  # Scale y: 6 grid units -> 30m

# Function to fit SpATS model and extract spatial effects
fit_spats_model <- function(data) {
  # Fit SpATS model using individual observations
  spats_model <- SpATS(response = "yield", 
                       spatial = ~ PSANOVA(x, y, nseg = c(10, 10)), 
                       genotype = "treatment", 
                       fixed = ~ 1,
                       data = data,
                       control = list(tolerance = 1e-03, monitoring = 0))
  
  # Create prediction grid
  x_seq <- seq(min(data$x), max(data$x), length.out = 50)
  y_seq <- seq(min(data$y), max(data$y), length.out = 50)
  pred_grid <- expand.grid(x = x_seq, y = y_seq)
  pred_grid$treatment <- "Control"  # Use Control as reference
  
  # Get spatial predictions
  spatial_pred <- predict(spats_model, pred_grid)
  pred_grid$spatial_effect <- spatial_pred$predicted.values
  
  # Scale coordinates to field layout
  pred_grid$x_scaled <- pred_grid$x * 5 - 2.5 # - 2.5 offset needed
  pred_grid$y_scaled <- pred_grid$y * 5 - 2.5 # - 2.5 offset needed
  
  return(list(model = spats_model, predictions = pred_grid))
}

# Function to fit variogram model and extract spatial effects
fit_variogram_model <- function(data) {
  # Create spatial points data frame
  coordinates(data) <- ~x+y
  
  # Fit linear model to get residuals
  lm_model <- lm(yield ~ treatment, data = data)
  data$residuals <- residuals(lm_model)
  
  # Calculate experimental variogram
  exp_variogram <- variogram(residuals ~ 1, data = data)
  
  # Fit theoretical variogram model - LINEAR MODEL
  vario_model <- fit.variogram(exp_variogram, model = vgm(psill = 1, model = "Lin", range = 3))
  
  # Create prediction grid
  x_seq <- seq(min(coordinates(data)[,1]), max(coordinates(data)[,1]), length.out = 50)
  y_seq <- seq(min(coordinates(data)[,2]), max(coordinates(data)[,2]), length.out = 50)
  pred_grid <- expand.grid(x = x_seq, y = y_seq)
  coordinates(pred_grid) <- ~x+y
  gridded(pred_grid) <- TRUE
  
  # Perform kriging
  kriging_result <- krige(residuals ~ 1, data, pred_grid, model = vario_model)
  
  # Convert back to data frame and scale coordinates
  pred_df <- as.data.frame(kriging_result)
  pred_df$x_scaled <- pred_df$x * 5 - 2.5 # - 2.5 offset needed
  pred_df$y_scaled <- pred_df$y * 5 - 2.5 # - 2.5 offset needed
  pred_df$spatial_effect <- pred_df$var1.pred
  
  return(list(
    lm_model = lm_model,
    exp_variogram = exp_variogram, 
    vario_model = vario_model,
    kriging_result = kriging_result,
    predictions = pred_df
  ))
}

# Function to calculate model comparison metrics
calculate_model_comparison <- function() {
  # True treatment effects (known from data generation)
  true_treatment_effects <- c("Control" = 0, "Test" = 1, "Reference" = 0.5)
  
  # Calculate true environmental gradient for each observation
  dataset1$x_norm <- (dataset1$x - 1) / (9 - 1)  # Normalize to 0-1
  dataset1$y_norm <- (dataset1$y - 1) / (6 - 1)  # Normalize to 0-1
  dataset1$diagonal_dist <- sqrt(dataset1$x_norm^2 + dataset1$y_norm^2) / sqrt(2)
  dataset1$true_env_effect <- -1.5 + 3 * dataset1$diagonal_dist  # True environmental gradient
  
  # Get true treatment effect for each observation
  dataset1$true_treatment_effect <- sapply(dataset1$treatment, function(x) true_treatment_effects[x])
  
  # ===== RCBD MODEL EVALUATION =====
  rcbd_treatment_errors_by_trt <- rep(NA, 3)
  names(rcbd_treatment_errors_by_trt) <- c("Control", "Test", "Reference")
  rcbd_assumption_tests <- list()
  
  if (exists("rcbd_model_global")) {
    # Ensure block_factor exists for predictions
    if (!"block_factor" %in% names(dataset1)) {
      dataset1$block_factor <- as.factor(dataset1$block)
    }
    
    # Get RCBD predictions
    dataset1$rcbd_pred <- predict(rcbd_model_global, dataset1)
    
    # Extract estimated treatment effects from RCBD model
    rcbd_coef <- coefficients(rcbd_model_global)
    
    # Handle case where some coefficients might be missing
    rcbd_treatment_est <- c("Control" = 0)  # Reference level
    
    if ("treatmentTest" %in% names(rcbd_coef)) {
      rcbd_treatment_est["Test"] <- rcbd_coef["treatmentTest"]
    } else {
      rcbd_treatment_est["Test"] <- 0
    }
    
    if ("treatmentReference" %in% names(rcbd_coef)) {
      rcbd_treatment_est["Reference"] <- rcbd_coef["treatmentReference"]
    } else {
      rcbd_treatment_est["Reference"] <- 0
    }
    
    # Calculate treatment effect errors by treatment
    rcbd_treatment_errors_by_trt <- abs(rcbd_treatment_est - true_treatment_effects)
    rcbd_mean_treatment_error <- mean(rcbd_treatment_errors_by_trt, na.rm = TRUE)
    
    # For RCBD, environmental effect is captured by block effects
    # Calculate residuals after removing treatment effects to get environmental component estimation
    dataset1$rcbd_treatment_component <- sapply(dataset1$treatment, function(x) rcbd_treatment_est[x])
    dataset1$rcbd_env_est <- dataset1$rcbd_pred - dataset1$rcbd_treatment_component - rcbd_coef["(Intercept)"]
    
    # Calculate environmental effect errors
    rcbd_env_errors <- abs(dataset1$rcbd_env_est - dataset1$true_env_effect)
    rcbd_mean_env_error <- mean(rcbd_env_errors, na.rm = TRUE)
    
    # Calculate environmental errors by treatment
    rcbd_env_errors_by_trt <- aggregate(rcbd_env_errors, 
                                       by = list(treatment = dataset1$treatment), 
                                       FUN = mean, na.rm = TRUE)
    names(rcbd_env_errors_by_trt) <- c("treatment", "env_error")
    
    # RCBD Assumption Tests
    rcbd_residuals <- residuals(rcbd_model_global)
    rcbd_fitted <- fitted(rcbd_model_global)
    
    # Normality test (Shapiro-Wilk)
    rcbd_assumption_tests$normality <- tryCatch({
      shapiro.test(rcbd_residuals)
    }, error = function(e) list(p.value = NA, method = "Shapiro-Wilk test failed"))
    
    # Homoscedasticity tests (Levene's and Bartlett's)
    rcbd_assumption_tests$levene <- tryCatch({
      library(car)
      result <- leveneTest(residuals(rcbd_model_global), dataset1$treatment)
      # Ensure the result has the expected structure
      if (is.data.frame(result) && "Pr(>F)" %in% names(result)) {
        result
      } else {
        list(`Pr(>F)` = c(NA), `F value` = c(NA), Df = c(NA, NA), method = "Levene's test structure error")
      }
    }, error = function(e) {
      list(`Pr(>F)` = c(NA), `F value` = c(NA), Df = c(NA, NA), method = "Levene's test failed")
    })
    
    rcbd_assumption_tests$bartlett <- tryCatch({
      bartlett.test(residuals(rcbd_model_global), dataset1$treatment)
    }, error = function(e) list(p.value = NA, statistic = NA, parameter = NA, method = "Bartlett's test failed"))
    
    # Independence test (Durbin-Watson)
    rcbd_assumption_tests$independence <- tryCatch({
      library(lmtest)
      dwtest(rcbd_model_global)
    }, error = function(e) list(p.value = NA, method = "Durbin-Watson test failed"))
    
  } else {
    rcbd_mean_treatment_error <- NA
    rcbd_mean_env_error <- NA
    rcbd_env_errors_by_trt <- data.frame(treatment = c("Control", "Test", "Reference"), 
                                        env_error = rep(NA, 3))
  }
  
  # ===== SpATS MODEL EVALUATION =====
  spats_treatment_errors_by_trt <- rep(NA, 3)
  names(spats_treatment_errors_by_trt) <- c("Control", "Test", "Reference")
  spats_assumption_tests <- list()
  
  if (exists("spats_results_global")) {
    spats_model <- spats_results_global$model
    
    # Get SpATS predictions for original data
    dataset1$spats_pred <- predict(spats_model, dataset1)$predicted.values
    
    # Extract estimated treatment effects from SpATS model - FIXED
    spats_coef <- spats_model$coeff
    cat("SpATS coefficients found:", names(spats_coef), "\n")
    
    # Handle SpATS coefficient extraction more robustly
    spats_treatment_est <- c("Control" = 0)  # Reference level
    
    # Look for treatment coefficients with different possible names
    test_coef_names <- c("treatmentTest", "genotypeTest", "Test")
    ref_coef_names <- c("treatmentReference", "genotypeReference", "Reference")
    
    spats_treatment_est["Test"] <- 0
    spats_treatment_est["Reference"] <- 0
    
    for (name in test_coef_names) {
      if (name %in% names(spats_coef)) {
        spats_treatment_est["Test"] <- spats_coef[name]
        break
      }
    }
    
    for (name in ref_coef_names) {
      if (name %in% names(spats_coef)) {
        spats_treatment_est["Reference"] <- spats_coef[name]
        break
      }
    }
    
    # Calculate treatment effect errors by treatment
    spats_treatment_errors_by_trt <- abs(spats_treatment_est - true_treatment_effects)
    spats_mean_treatment_error <- mean(spats_treatment_errors_by_trt, na.rm = TRUE)
    
    # For SpATS, environmental effect is captured by spatial component
    # Extract spatial component (predicted - treatment effect - intercept)
    dataset1$spats_treatment_component <- sapply(dataset1$treatment, function(x) spats_treatment_est[x])
    
    # Handle intercept extraction
    intercept_names <- c("(Intercept)", "intercept", "Intercept")
    intercept_val <- 0
    for (name in intercept_names) {
      if (name %in% names(spats_coef)) {
        intercept_val <- spats_coef[name]
        break
      }
    }
    
    dataset1$spats_env_est <- dataset1$spats_pred - dataset1$spats_treatment_component - intercept_val
    
    # Calculate environmental effect errors (handle potential NaN/Inf values)
    spats_env_errors <- abs(dataset1$spats_env_est - dataset1$true_env_effect)
    spats_env_errors_clean <- spats_env_errors[is.finite(spats_env_errors)]  # Remove non-finite values
    spats_mean_env_error <- if(length(spats_env_errors_clean) > 0) mean(spats_env_errors_clean, na.rm = TRUE) else NA
    
    # Calculate environmental errors by treatment (with finite value handling)
    dataset1$spats_env_errors_finite <- ifelse(is.finite(spats_env_errors), spats_env_errors, NA)
    spats_env_errors_by_trt <- aggregate(dataset1$spats_env_errors_finite, 
                                        by = list(treatment = dataset1$treatment), 
                                        FUN = function(x) mean(x, na.rm = TRUE))
    names(spats_env_errors_by_trt) <- c("treatment", "env_error")
    
    # SpATS Assumption Tests
    spats_residuals <- residuals(spats_model)
    spats_fitted <- fitted(spats_model)
    
    # Normality test
    spats_assumption_tests$normality <- tryCatch({
      if (length(spats_residuals) > 3 && length(spats_residuals) <= 5000) {
        shapiro.test(spats_residuals)
      } else {
        list(p.value = NA, method = "Sample size inappropriate for Shapiro-Wilk")
      }
    }, error = function(e) list(p.value = NA, method = "Shapiro-Wilk test failed"))
    
    # For SpATS, we use different tests as it's a more complex model
    spats_assumption_tests$spatial_correlation <- tryCatch({
      # Test for remaining spatial correlation in residuals
      library(gstat)
      coords_data <- dataset1[, c("x", "y")]
      resid_data <- data.frame(coords_data, residuals = spats_residuals)
      coordinates(resid_data) <- ~x+y
      variogram_resid <- variogram(residuals ~ 1, resid_data)
      # Simple check: if initial nugget is much smaller than sill, there's spatial correlation
      if (nrow(variogram_resid) > 0) {
        initial_gamma <- variogram_resid$gamma[1]
        max_gamma <- max(variogram_resid$gamma)
        list(spatial_correlation_ratio = initial_gamma / max_gamma, 
             method = "Residual variogram analysis")
      } else {
        list(spatial_correlation_ratio = NA, method = "Variogram calculation failed")
      }
    }, error = function(e) list(spatial_correlation_ratio = NA, method = "Spatial correlation test failed"))
    
  } else {
    spats_mean_treatment_error <- NA
    spats_mean_env_error <- NA
    spats_env_errors_by_trt <- data.frame(treatment = c("Control", "Test", "Reference"), 
                                         env_error = rep(NA, 3))
  }
  
  # Create detailed comparison tables
  overall_comparison <- data.frame(
    Model = c("RCBD", "SpATS"),
    Treatment_Error_t_ha = c(rcbd_mean_treatment_error, spats_mean_treatment_error),
    Environmental_Error_t_ha = c(rcbd_mean_env_error, spats_mean_env_error)
  )
  
  # Treatment-specific errors - FIXED indexing
  treatment_comparison <- data.frame(
    Treatment = c("Control", "Test", "Reference"),
    RCBD_Treatment_Error = rcbd_treatment_errors_by_trt,
    SpATS_Treatment_Error = spats_treatment_errors_by_trt,
    RCBD_Environmental_Error = rcbd_env_errors_by_trt$env_error[1:3],
    SpATS_Environmental_Error = spats_env_errors_by_trt$env_error[1:3]
  )
  
  # Write to file
  sink("model_comparison_summary.txt")
  cat("=== MODEL COMPARISON SUMMARY ===\n")
  cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  cat("Comparison of RCBD vs SpATS Model Performance\n")
  cat("Mean Absolute Errors (MAE) in t/ha\n\n")
  
  # Overall comparison table
  cat("OVERALL MODEL PERFORMANCE:\n")
  cat("==========================\n\n")
  cat(sprintf("%-15s %-20s %-25s\n", 
              "Model", 
              "Treatment_Error_t_ha", 
              "Environmental_Error_t_ha"))
  cat(sprintf("%-15s %-20s %-25s\n", 
              "-----", 
              "--------------------", 
              "-------------------------"))
  
  for(i in 1:nrow(overall_comparison)) {
    cat(sprintf("%-15s %-20.4f %-25.4f\n", 
                overall_comparison$Model[i], 
                overall_comparison$Treatment_Error_t_ha[i], 
                overall_comparison$Environmental_Error_t_ha[i]))
  }
  
  # Treatment-specific comparison table
  cat("\n\nTREATMENT-SPECIFIC PERFORMANCE:\n")
  cat("===============================\n\n")
  cat(sprintf("%-12s %-18s %-18s %-22s %-22s\n", 
              "Treatment", 
              "RCBD_Treat_Error", 
              "SpATS_Treat_Error",
              "RCBD_Environ_Error",
              "SpATS_Environ_Error"))
  cat(sprintf("%-12s %-18s %-18s %-22s %-22s\n", 
              "---------", 
              "-----------------", 
              "-----------------",
              "---------------------",
              "---------------------"))
  
  for(i in 1:nrow(treatment_comparison)) {
    cat(sprintf("%-12s %-18.4f %-18.4f %-22.4f %-22.4f\n", 
                treatment_comparison$Treatment[i], 
                treatment_comparison$RCBD_Treatment_Error[i], 
                treatment_comparison$SpATS_Treatment_Error[i],
                treatment_comparison$RCBD_Environmental_Error[i],
                treatment_comparison$SpATS_Environmental_Error[i]))
  }
  
  # Assumption Tests Results
  cat("\n\nASSUMPTION TESTS RESULTS:\n")
  cat("=========================\n\n")
  
  # RCBD Assumption Tests
  cat("RCBD Model Assumptions:\n")
  cat("-----------------------\n")
  if (exists("rcbd_model_global")) {
    cat("1. Normality of Residuals (Shapiro-Wilk Test):\n")
    if (!is.null(rcbd_assumption_tests$normality) && !is.na(rcbd_assumption_tests$normality$p.value)) {
      cat(sprintf("   W = %.4f, p-value = %.4f\n", 
                  rcbd_assumption_tests$normality$statistic, 
                  rcbd_assumption_tests$normality$p.value))
      cat(sprintf("   Interpretation: %s\n", 
                  ifelse(rcbd_assumption_tests$normality$p.value > 0.05, 
                         "Residuals are normally distributed (p > 0.05)", 
                         "Residuals are NOT normally distributed (p <= 0.05)")))
    } else {
      cat("   Test failed or not applicable\n")
    }
    
    cat("\n2. Homoscedasticity - Levene's Test (robust to non-normality):\n")
    if (!is.null(rcbd_assumption_tests$levene) && 
        is.list(rcbd_assumption_tests$levene) &&
        "`Pr(>F)`" %in% names(rcbd_assumption_tests$levene) &&
        !is.na(rcbd_assumption_tests$levene$`Pr(>F)`[1])) {
      cat(sprintf("   F = %.4f, df = %d, %d, p-value = %.4f\n", 
                  rcbd_assumption_tests$levene$`F value`[1],
                  rcbd_assumption_tests$levene$Df[1],
                  rcbd_assumption_tests$levene$Df[2],
                  rcbd_assumption_tests$levene$`Pr(>F)`[1]))
      cat(sprintf("   Interpretation: %s\n", 
                  ifelse(rcbd_assumption_tests$levene$`Pr(>F)`[1] > 0.05, 
                         "Homoscedasticity assumption met (p > 0.05)", 
                         "Heteroscedasticity detected (p <= 0.05)")))
    } else {
      cat("   Test failed or not applicable\n")
    }
    
    cat("\n3. Homoscedasticity - Bartlett's Test (sensitive to normality):\n")
    if (!is.null(rcbd_assumption_tests$bartlett) && 
        is.list(rcbd_assumption_tests$bartlett) &&
        "p.value" %in% names(rcbd_assumption_tests$bartlett) &&
        !is.na(rcbd_assumption_tests$bartlett$p.value)) {
      cat(sprintf("   K-squared = %.4f, df = %d, p-value = %.4f\n", 
                  rcbd_assumption_tests$bartlett$statistic, 
                  rcbd_assumption_tests$bartlett$parameter,
                  rcbd_assumption_tests$bartlett$p.value))
      cat(sprintf("   Interpretation: %s\n", 
                  ifelse(rcbd_assumption_tests$bartlett$p.value > 0.05, 
                         "Homoscedasticity assumption met (p > 0.05)", 
                         "Heteroscedasticity detected (p <= 0.05)")))
    } else {
      cat("   Test failed or not applicable\n")
    }
    
    cat("\n4. Independence (Durbin-Watson Test):\n")
    if (!is.null(rcbd_assumption_tests$independence) && !is.na(rcbd_assumption_tests$independence$p.value)) {
      cat(sprintf("   DW = %.4f, p-value = %.4f\n", 
                  rcbd_assumption_tests$independence$statistic, 
                  rcbd_assumption_tests$independence$p.value))
      cat(sprintf("   Interpretation: %s\n", 
                  ifelse(rcbd_assumption_tests$independence$p.value > 0.05, 
                         "No autocorrelation detected (p > 0.05)", 
                         "Autocorrelation detected (p <= 0.05)")))
    } else {
      cat("   Test failed or not applicable\n")
    }
    
    # Additional diagnostics if homoscedasticity is violated
    homoscedasticity_violated <- FALSE
    if (!is.null(rcbd_assumption_tests$levene) && 
        is.list(rcbd_assumption_tests$levene) &&
        "`Pr(>F)`" %in% names(rcbd_assumption_tests$levene) &&
        !is.na(rcbd_assumption_tests$levene$`Pr(>F)`[1])) {
      if (rcbd_assumption_tests$levene$`Pr(>F)`[1] <= 0.05) homoscedasticity_violated <- TRUE
    }
    if (!is.null(rcbd_assumption_tests$bartlett) && 
        is.list(rcbd_assumption_tests$bartlett) &&
        "p.value" %in% names(rcbd_assumption_tests$bartlett) &&
        !is.na(rcbd_assumption_tests$bartlett$p.value)) {
      if (rcbd_assumption_tests$bartlett$p.value <= 0.05) homoscedasticity_violated <- TRUE
    }
    
    if (homoscedasticity_violated) {
      cat("\n   HETEROSCEDASTICITY DETECTED - Additional Diagnostics:\n")
      cat("   =====================================================\n")
      
      # Variance by treatment group
      cat("   Variance by Treatment Group:\n")
      treatment_vars <- aggregate(residuals(rcbd_model_global)^2, 
                                 by = list(treatment = dataset1$treatment), 
                                 FUN = mean, na.rm = TRUE)
      names(treatment_vars) <- c("Treatment", "Residual_Variance")
      for(i in 1:nrow(treatment_vars)) {
        cat(sprintf("   - %s: %.4f\n", treatment_vars$Treatment[i], treatment_vars$Residual_Variance[i]))
      }
      
      # Variance by block
      cat("   Variance by Block:\n")
      block_vars <- aggregate(residuals(rcbd_model_global)^2, 
                             by = list(block = dataset1$block), 
                             FUN = mean, na.rm = TRUE)
      names(block_vars) <- c("Block", "Residual_Variance")
      for(i in 1:nrow(block_vars)) {
        cat(sprintf("   - Block %d: %.4f\n", block_vars$Block[i], block_vars$Residual_Variance[i]))
      }
      
      # Recommendations
      cat("\n   Recommendations for Heteroscedasticity:\n")
      cat("   - Consider data transformation (log, sqrt, Box-Cox)\n")
      cat("   - Use robust standard errors (sandwich estimators)\n")
      cat("   - Consider weighted least squares if variance pattern is known\n")
      cat("   - Use generalized least squares with variance modeling\n")
    }
  } else {
    cat("RCBD model not available for assumption testing\n")
  }
  
  # SpATS Assumption Tests
  cat("\n\nSpATS Model Assumptions:\n")
  cat("------------------------\n")
  if (exists("spats_results_global")) {
    cat("1. Normality of Residuals (Shapiro-Wilk Test):\n")
    if (!is.null(spats_assumption_tests$normality) && !is.na(spats_assumption_tests$normality$p.value)) {
      cat(sprintf("   W = %.4f, p-value = %.4f\n", 
                  spats_assumption_tests$normality$statistic, 
                  spats_assumption_tests$normality$p.value))
      cat(sprintf("   Interpretation: %s\n", 
                  ifelse(spats_assumption_tests$normality$p.value > 0.05, 
                         "Residuals are normally distributed (p > 0.05)", 
                         "Residuals are NOT normally distributed (p <= 0.05)")))
    } else {
      cat("   Test failed or not applicable\n")
    }
    
    cat("\n2. Spatial Independence (Residual Variogram Analysis):\n")
    if (!is.null(spats_assumption_tests$spatial_correlation) && 
        !is.na(spats_assumption_tests$spatial_correlation$spatial_correlation_ratio)) {
      ratio <- spats_assumption_tests$spatial_correlation$spatial_correlation_ratio
      cat(sprintf("   Initial/Max Variogram Ratio = %.4f\n", ratio))
      cat(sprintf("   Interpretation: %s\n", 
                  ifelse(ratio < 0.1, 
                         "Good spatial modeling - little residual spatial correlation", 
                         "Some residual spatial correlation remains")))
    } else {
      cat("   Test failed or not applicable\n")
    }
  } else {
    cat("SpATS model not available for assumption testing\n")
  }
  
  cat("\n\nColumn Definitions:\n")
  cat("- Treatment_Error_t_ha: Mean absolute error between estimated and true treatment effects\n")
  cat("- Environmental_Error_t_ha: Mean absolute error between estimated and true environmental/spatial effects\n")
  cat("- RCBD_Treat_Error: Treatment-specific error for RCBD model\n")
  cat("- SpATS_Treat_Error: Treatment-specific error for SpATS model\n")
  cat("- RCBD_Environ_Error: Environmental estimation error by treatment for RCBD\n")
  cat("- SpATS_Environ_Error: Environmental estimation error by treatment for SpATS\n")
  
  cat("\n\nModel Descriptions:\n")
  cat("- RCBD: Randomized Complete Block Design with block effects capturing environmental variation\n")
  cat("- SpATS: Spatial Analysis of Field Trials with spline-based spatial modeling\n")
  
  cat("\n\nTrue Treatment Effects Used:\n")
  cat("- Control: 0.0 t/ha\n")
  cat("- Test: 1.0 t/ha\n") 
  cat("- Reference: 0.5 t/ha\n")
  
  cat("\n\nTrue Environmental Effect:\n")
  cat("- Diagonal gradient from -1.5 to +1.5 t/ha (bottom-left to top-right)\n")
  
  # Model performance interpretation
  cat("\n\nPERFORMANCE INTERPRETATION:\n")
  cat("===========================\n")
  cat("Lower values indicate better model performance (smaller errors)\n\n")
  
  if (!is.na(rcbd_mean_treatment_error) && !is.na(spats_mean_treatment_error)) {
    if (spats_mean_treatment_error < rcbd_mean_treatment_error) {
      cat("Treatment Effects: SpATS performs better than RCBD\n")
    } else {
      cat("Treatment Effects: RCBD performs better than SpATS\n")
    }
  }
  
  if (!is.na(rcbd_mean_env_error) && !is.na(spats_mean_env_error)) {
    if (spats_mean_env_error < rcbd_mean_env_error) {
      cat("Environmental Effects: SpATS performs better than RCBD\n")
    } else {
      cat("Environmental Effects: RCBD performs better than SpATS\n")
    }
  }
  
  sink()
  
  cat("Model comparison summary written to: model_comparison_summary.txt\n")
  
  return(list(overall = overall_comparison, by_treatment = treatment_comparison))
}

# Function to write model summaries to text files
write_model_summaries <- function(rcbd_model = NULL, variogram_results = NULL, spats_results = NULL) {
  
  # RCBD model summary
  if (!is.null(rcbd_model)) {
    sink("model_summaries_RCBD.txt")
    cat("=== RCBD MODEL SUMMARY ===\n")
    cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
    cat("Linear Model Summary:\n")
    cat("Formula: yield ~ treatment + block\n\n")
    print(summary(rcbd_model))
    cat("\n\nANOVA:\n")
    print(anova(rcbd_model))
    cat("\n\nTreatment Effects:\n")
    print(coefficients(rcbd_model))
    sink()
  }
  
  # Variogram model summary
  if (!is.null(variogram_results)) {
    sink("model_summaries_Variogram.txt")
    cat("=== VARIOGRAM MODEL SUMMARY ===\n")
    cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
    cat("Linear Model (for residuals):\n")
    print(summary(variogram_results$lm_model))
    cat("\n\nExperimental Variogram:\n")
    print(variogram_results$exp_variogram)
    cat("\n\nFitted Variogram Model:\n")
    print(variogram_results$vario_model)
    cat("\n\nVariogram Parameters:\n")
    cat("Model:", variogram_results$vario_model$model[2], "\n")
    cat("Nugget:", variogram_results$vario_model$psill[1], "\n")
    cat("Sill:", sum(variogram_results$vario_model$psill), "\n")
    cat("Range:", variogram_results$vario_model$range[2], "\n")
    sink()
  }
  
  # SpATS model summary
  if (!is.null(spats_results)) {
    sink("model_summaries_SpATS.txt")
    cat("=== SpATS MODEL SUMMARY ===\n")
    cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
    print(summary(spats_results$model))
    cat("\n\nModel Information:\n")
    cat("Response variable:", spats_results$model$model$response, "\n")
    cat("Spatial terms:", "PSANOVA(x, y)", "\n")
    cat("Fixed effects:", "treatment + intercept", "\n")
    cat("Effective dimensions:", spats_results$model$eff.dim, "\n")
    cat("Deviance:", spats_results$model$deviance, "\n")
    cat("AIC:", spats_results$model$aic, "\n")
    sink()
  }
  
  cat("Model summaries written to:\n")
  cat("- model_summaries_RCBD.txt\n")
  cat("- model_summaries_Variogram.txt\n") 
  cat("- model_summaries_SpATS.txt\n")
}

# Function to add corn cobs with integrated ggplot legend using individual observations
add_wheat_spikes_with_legend <- function(plot_obj, data) {
  # Use individual observations data directly
  individual_data <- data
  
  # Scale coordinates to match field layout (same scaling as dataset2)
  # Original grid coordinates -> Field coordinates: multiply by 5
  individual_data$x_scaled <- individual_data$x * 5 - 2.5  # Scale x to field coordinates
  individual_data$y_scaled <- individual_data$y * 5 - 2.5  # Scale y to field coordinates
  
  # Define yield categories based on quantiles of individual yields
  individual_data$yield_category <- cut(individual_data$yield, 
                                       breaks = c(-Inf, quantile(individual_data$yield, 0.33), 
                                                 quantile(individual_data$yield, 0.67), Inf),
                                       labels = c("Low", "Medium", "High"))
  
  # Define corresponding sizes for each category
  size_mapping <- c("Low" = 3, "Medium" = 5, "High" = 7)
  individual_data$spike_size <- size_mapping[individual_data$yield_category]
  
  # Add wheat spike emoji annotations for each individual observation using scaled coordinates
  for(i in 1:nrow(individual_data)) {
    plot_obj <- plot_obj + 
      annotate("text", x = individual_data$x_scaled[i], y = individual_data$y_scaled[i], 
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

# Base plot function
create_base_plot <- function() {
  ggplot() +
    # Plot boundaries
    geom_rect(data = plot_coords, 
              aes(xmin = x_center - 7.5, xmax = x_center + 7.5,
                  ymin = y_center - 5, ymax = y_center + 5,
                  fill = treatment), 
              color = "black", alpha = 0.3, linewidth = 1) +
    scale_fill_manual(values = treatment_colors, name = "Treatment") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "#000000", linewidth = 0.5),
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
  # Fit RCBD linear model
  dataset1$block_factor <- as.factor(dataset1$block)
  rcbd_model <- lm(yield ~ treatment + block_factor, data = dataset1)
  
  # Store model for summary output
  assign("rcbd_model_global", rcbd_model, envir = .GlobalEnv)
  
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
  
  # Map plot_coords treatments to dataset2 treatment values
  # Convert plot_coords treatment codes to full names for matching
  plot_coords_extended <- plot_coords
  plot_coords_extended$treatment_full <- sapply(plot_coords_extended$treatment, function(x) {
    switch(x, "T" = "Test", "C" = "Control", "R" = "Reference")
  })
  plot_coords_extended$block_num <- sapply(plot_coords_extended$block, function(x) {
    switch(x, "Block 1" = 1, "Block 2" = 2, "Block 3" = 3)
  })
  
  # Add true_treatment_effect manually to avoid join issues
  plot_coords_extended$true_treatment_effect <- sapply(plot_coords_extended$treatment_full, function(x) {
    switch(x, "Control" = 0, "Reference" = 0.5, "Test" = 1)
  })
  
  # Block labels data (positioned to the left of the plot)
  block_labels <- data.frame(
    block = c("Block 1", "Block 2", "Block 3"),
    x_pos = rep(-5, 3),  # Left of the plot area
    y_pos = c(25, 15, 5)  # Middle height of each block row
  )
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "Enviromental\nSpatial Effect\n(t/ha)") +
    # Plot rectangles with border color based on treatment effect (with gap to avoid overlap)
    geom_rect(data = plot_coords_extended, 
              aes(xmin = x_center - 7.4, xmax = x_center + 7.4,
                  ymin = y_center - 4.9, ymax = y_center + 4.9,
                  color = true_treatment_effect), 
              fill = NA, linewidth = 3, alpha = 0.8) +
    scale_color_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F",
                         midpoint = 0.5,  # Reference treatment at midpoint
                         name = "Treatment\nEffect (t/ha)",
                         breaks = c(0, 0.5, 1),
                         labels = c("Control (0)", "Reference (0.5)", "Test (1)")) +
    # Block labels on the left side
    geom_text(data = block_labels,
              aes(x = x_pos, y = y_pos, label = block),
              size = 5, fontface = "bold", hjust = 1, color = "black") +
    # Treatment labels
    geom_text(data = plot_coords,
              aes(x = x_center, y = y_center, label = treatment),
              size = 13, fontface = "bold", color = "black") +
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(-8, 45) + ylim(0, 30) +  # Extended x-axis to accommodate block labels
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "RCBD: Environmental Gradient vs Block Effects") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  # Add corn cobs with ggplot legend
  p <- add_wheat_spikes_with_legend(p, dataset1)
  
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
  
  # Fit variogram model to get actual spatial effects
  cat("Fitting variogram model for spatial effects...\n")
  variogram_results <- fit_variogram_model(dataset1)
  spatial_data <- variogram_results$predictions
  
  # Store results for summary output
  assign("variogram_results_global", variogram_results, envir = .GlobalEnv)
  
  p <- ggplot() +
    # Environmental gradient background
    geom_raster(data = grid_data, aes(x = x, y = y, fill = env_gradient), alpha = 0.6) +
    scale_fill_gradient2(low = "#FFF5EB", mid = "#FDAE61", high = "#D7301F", 
                        midpoint = 0, name = "Environmental\nSpatial Effect\n(t/ha)") +
    # Actual spatial effect contours from variogram model  
    geom_contour(data = spatial_data[is.finite(spatial_data$spatial_effect), ], 
                aes(x = x_scaled, y = y_scaled, z = spatial_effect), 
                color = "blue", linewidth = 1.5, alpha = 0.8) +
    # Spatial effect contour labels (only for finite values)
    geom_text_contour(data = spatial_data[is.finite(spatial_data$spatial_effect), ], 
                     aes(x = x_scaled, y = y_scaled, z = spatial_effect),
                     color = "darkblue", size = 4, fontface = "bold") +
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
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "Variogram: Environmental Gradient vs Kriging Spatial Effects") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    ) +
    # Add legend for spatial effect
    annotate("text", x = 35, y = 28, label = "Blue contours:\nVariogram kriging\nspatial effects", 
             size = 4, hjust = 0, color = "darkblue", fontface = "bold")
  
  # Add wheat spikes with ggplot legend
  p <- add_wheat_spikes_with_legend(p, dataset1)
  
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
  # Fit SpATS model with finer spatial resolution for spline effects
  cat("Fitting SpATS model with spline effects...\n")
  spats_model <- SpATS(response = "yield", 
                       spatial = ~ PSANOVA(x, y, nseg = c(15, 15)), 
                       genotype = "treatment", 
                       fixed = ~ 1,
                       data = dataset1,
                       control = list(tolerance = 1e-03, monitoring = 0))
  
  # Create finer prediction grid for smoother splines
  x_seq_fine <- seq(min(dataset1$x), max(dataset1$x), length.out = 60)
  y_seq_fine <- seq(min(dataset1$y), max(dataset1$y), length.out = 60)
  pred_grid_fine <- expand.grid(x = x_seq_fine, y = y_seq_fine)
  pred_grid_fine$treatment <- "Control"  # Use Control as reference
  
  # Get spatial predictions
  spatial_pred <- predict(spats_model, pred_grid_fine)
  pred_grid_fine$spline_effect <- spatial_pred$predicted.values
  
  # Scale coordinates to field layout
  pred_grid_fine$x_scaled <- pred_grid_fine$x * 5 - 2.5 # - 2.5 offset needed
  pred_grid_fine$y_scaled <- pred_grid_fine$y * 5 - 2.5 # - 2.5 offset needed
  
  # Store results for summary output
  spats_results <- list(model = spats_model, predictions = pred_grid_fine)
  assign("spats_results_global", spats_results, envir = .GlobalEnv)
  
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
    # Coordinate system
    coord_fixed(ratio = 1) +
    xlim(0, 45) + ylim(0, 30) +
    labs(x = "X coordinate (m)", y = "Y coordinate (m)",
         title = "SpATS: Environmental Gradient vs Estimated Spline Effects") +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    ) +
    # Add legend for spline effect
    annotate("text", x = 35, y = 28, label = "Purple contours:\nSpATS estimated\nspline effects", 
             size = 4, hjust = 0, color = "purple4", fontface = "bold")
  
  # Add wheat spikes with ggplot legend
  p <- add_wheat_spikes_with_legend(p, dataset1)
  
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

# Write model summaries to text files
cat("\nWriting model summaries...\n")
write_model_summaries(
  rcbd_model = if(exists("rcbd_model_global")) rcbd_model_global else NULL,
  variogram_results = if(exists("variogram_results_global")) variogram_results_global else NULL,
  spats_results = if(exists("spats_results_global")) spats_results_global else NULL
)

# Calculate and write model comparison
cat("\nCalculating model comparison metrics...\n")
comparison_results <- calculate_model_comparison()
