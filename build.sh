#!/bin/bash

# PhD Thesis Build Script
# This script generates all required data and compiles the thesis, presentation, and research papers

set -e  # Exit on any error

echo "=========================================="
echo "PhD Thesis Build Script"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo "Checking required tools..."
if ! command_exists R; then
    echo "Error: R is not installed or not in PATH"
    exit 1
fi

if ! command_exists pdflatex; then
    echo "Error: pdflatex is not installed or not in PATH"
    exit 1
fi

if ! command_exists bibtex; then
    echo "Error: bibtex is not installed or not in PATH"
    exit 1
fi

if ! command_exists python3; then
    echo "Warning: python3 is not installed or not in PATH. Jupyter notebook execution will be skipped."
    PYTHON_AVAILABLE=false
else
    PYTHON_AVAILABLE=true
fi

echo "All required tools found."
echo ""

# Step 1: Generate data and plots for all research components
echo "Step 1: Installing R packages and generating data..."
echo "----------------------------------------"

cd Presentazione

# Install R packages
echo "Installing R packages..."
Rscript install_packages.R

# Generate trial data
echo "Generating trial data..."
Rscript generate_trial_data.R

# Generate integrated analysis plots
echo "Generating integrated analysis plots (regular)..."
Rscript integrated_trial_analysis.R

echo "Generating integrated analysis plots (irregular)..."
Rscript integrated_trial_analysis_irregular.R

# Copy generated files to main directory if needed
cp dataset1_detailed.csv ../
cp dataset2_plots.csv ../
cp trial_models.RData ../
cp integrated_rcbd_spats_comparison.png ../
cp integrated_rcbd_spats_comparison_irregular.png ../

cd ..

echo "Data generation completed."
echo ""

# Step 1.5: Generate plots for anomaly detection research (if Python available)
echo "Step 1.5: Generating anomaly detection plots..."
echo "----------------------------------------"

if [ "$PYTHON_AVAILABLE" = true ]; then
    cd Anomaly_detection
    
    echo "Running anomaly detection analysis..."
    
    # Try using the custom Python script first
    if [ -f "generate_plots.py" ]; then
        echo "Running custom plot generation script..."
        python3 generate_plots.py 2>/dev/null || {
            echo "Warning: Custom plot script failed. Trying Jupyter notebook..."
        }
    fi
    
    # If custom script failed or doesn't exist, try Jupyter
    if command_exists jupyter && [ -f "Results.ipynb" ]; then
        echo "Converting and executing Jupyter notebook..."
        jupyter nbconvert --to script Results.ipynb --output temp_results.py 2>/dev/null || {
            echo "Warning: Jupyter nbconvert failed."
        }
        
        if [ -f temp_results.py ]; then
            python3 temp_results.py 2>/dev/null || {
                echo "Warning: Python script execution failed."
            }
            rm -f temp_results.py
        fi
    fi
    
    # Check if required plot files exist
    required_plots=("Plant_Pathology_Dataset_Anomaly_Detection_Performance.pdf" 
                   "Plant_Village_Dataset_Anomaly_Detection_Performance.pdf"
                   "Plant_Pathology_Dataset_Clustering_Performance.pdf"
                   "Plant_Village_Dataset_Clustering_Performance.pdf")
    
    for plot in "${required_plots[@]}"; do
        if [ ! -f "$plot" ]; then
            echo "Warning: $plot missing. LaTeX compilation may fail if this file is referenced."
        else
            echo "✓ Found: $plot"
        fi
    done
    
    cd ..
else
    echo "Python not available. Using existing anomaly detection plots..."
fi

echo "Anomaly detection plot generation completed."
echo ""

# Step 2: Compile the research papers
echo "Step 2: Compiling research papers..."
echo "----------------------------------------"

# Compile Plant Count paper
echo "Compiling Plant Count paper (remotesensing-3622397.tex)..."
cd Plant_count
echo "Running pdflatex (1st pass)..."
pdflatex -interaction=nonstopmode remotesensing-3622397.tex
echo "Running bibtex..."
bibtex remotesensing-3622397 || echo "Warning: bibtex failed for Plant Count paper (may use embedded bibliography)"
echo "Running pdflatex (2nd pass)..."
pdflatex -interaction=nonstopmode remotesensing-3622397.tex
echo "Running pdflatex (3rd pass)..."
pdflatex -interaction=nonstopmode remotesensing-3622397.tex
echo "Plant Count paper compilation completed: Plant_count/remotesensing-3622397.pdf"
cd ..

# Compile Phytotoxicity Score paper
echo "Compiling Phytotoxicity Score paper (Phytoxicity_score.tex)..."
cd Phytotoxicity_score
echo "Running pdflatex (1st pass)..."
pdflatex -interaction=nonstopmode Phytoxicity_score.tex
echo "Running bibtex..."
bibtex Phytoxicity_score
echo "Running pdflatex (2nd pass)..."
pdflatex -interaction=nonstopmode Phytoxicity_score.tex
echo "Running pdflatex (3rd pass)..."
pdflatex -interaction=nonstopmode Phytoxicity_score.tex
echo "Phytotoxicity Score paper compilation completed: Phytotoxicity_score/Phytoxicity_score.pdf"
cd ..

# Compile Anomaly Detection paper
echo "Compiling Anomaly Detection paper (Anomaly_detection.tex)..."
cd Anomaly_detection
echo "Running pdflatex (1st pass)..."
pdflatex -interaction=nonstopmode Anomaly_detection.tex
echo "Running bibtex..."
bibtex Anomaly_detection
echo "Running pdflatex (2nd pass)..."
pdflatex -interaction=nonstopmode Anomaly_detection.tex
echo "Running pdflatex (3rd pass)..."
pdflatex -interaction=nonstopmode Anomaly_detection.tex
echo "Anomaly Detection paper compilation completed: Anomaly_detection/Anomaly_detection.pdf"
cd ..

echo "All research papers compilation completed."
echo ""

# Step 3: Compile the main thesis
echo "Step 3: Compiling main thesis (Thesis.tex)..."
echo "----------------------------------------"

# First pass
echo "Running pdflatex (1st pass)..."
pdflatex -interaction=nonstopmode Thesis.tex

# Run bibtex for bibliography
echo "Running bibtex..."
bibtex Thesis

# Second pass for cross-references
echo "Running pdflatex (2nd pass)..."
pdflatex -interaction=nonstopmode Thesis.tex

# Third pass to resolve all references
echo "Running pdflatex (3rd pass)..."
pdflatex -interaction=nonstopmode Thesis.tex

echo "Main thesis compilation completed: Thesis.pdf"
echo ""

# Step 4: Compile the presentation
echo "Step 4: Compiling presentation..."
echo "----------------------------------------"

cd Presentazione

# First pass
echo "Running pdflatex for presentation (1st pass)..."
pdflatex -interaction=nonstopmode PhD_Thesis_Presentation.tex

# Second pass for references
echo "Running pdflatex for presentation (2nd pass)..."
pdflatex -interaction=nonstopmode PhD_Thesis_Presentation.tex

cd ..

echo "Presentation compilation completed: Presentazione/PhD_Thesis_Presentation.pdf"
echo ""

# Step 5: Clean up auxiliary files
echo "Step 5: Cleaning up auxiliary files..."
echo "----------------------------------------"

# Clean main directory
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz texput.log

# Clean presentation directory
cd Presentazione
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.bbl *.blg *.synctex.gz texput.log
cd ..

# Clean research paper directories
cd Plant_count
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz texput.log
cd ..

cd Phytotoxicity_score
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz texput.log
cd ..

cd Anomaly_detection
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz texput.log
cd ..

echo "Cleanup completed."
echo ""

# Step 6: Summary
echo "=========================================="
echo "BUILD COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Generated files:"
echo "- Thesis.pdf (main thesis document)"
echo "- Presentazione/PhD_Thesis_Presentation.pdf (presentation)"
echo ""
echo "Research papers:"
echo "- Plant_count/remotesensing-3622397.pdf (Plant counting research)"
echo "- Phytotoxicity_score/Phytoxicity_score.pdf (Phytotoxicity scoring research)"
echo "- Anomaly_detection/Anomaly_detection.pdf (Anomaly detection research)"
echo ""
echo "Supporting data files:"
echo "- dataset1_detailed.csv"
echo "- dataset2_plots.csv" 
echo "- trial_models.RData"
echo "- integrated_rcbd_spats_comparison.png"
echo "- integrated_rcbd_spats_comparison_irregular.png"
echo ""
echo "Plots and figures:"
echo "- Anomaly_detection/Plant_Pathology_Dataset_Anomaly_Detection_Performance.pdf"
echo "- Anomaly_detection/Plant_Village_Dataset_Anomaly_Detection_Performance.pdf"
echo "- Anomaly_detection/Plant_Pathology_Dataset_Clustering_Performance.pdf"
echo "- Anomaly_detection/Plant_Village_Dataset_Clustering_Performance.pdf"
echo ""
echo "All files are ready for submission!"
