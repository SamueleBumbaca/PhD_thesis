# PhD Thesis: Geomatic Techniques to Support Phytosanitary Products Tests within the EPPO Standard Framework

This repository contains the complete PhD thesis by Samuele Bumbaca, including the main thesis document, presentation, and three research papers.

## Repository Structure

```
├── build.sh                              # Main build script
├── Thesis.tex                           # Main thesis document
├── Phd_Thesis_SBumbaca.bib             # Main bibliography
├── Intestazione/                         # Title page
│   └── t3._Thesis_first_page.pdf
├── Presentazione/                        # Thesis presentation
│   ├── PhD_Thesis_Presentation.tex
│   ├── Imgs/                            # Presentation images
│   ├── install_packages.R              # R package installation
│   ├── generate_trial_data.R           # Trial data generation
│   ├── integrated_trial_analysis.R     # Regular gradient analysis
│   └── integrated_trial_analysis_irregular.R  # Irregular gradient analysis
├── Plant_count/                          # Research Paper 1: Plant Counting
│   ├── remotesensing-3622397.tex
│   ├── remotesensing-3622397.bib
│   └── Unsupervised_plant_counting.bib
├── Phytotoxicity_score/                  # Research Paper 2: Phytotoxicity Scoring
│   ├── Phytoxicity_score.tex
│   ├── SAGEA_DiSAFA.bib
│   └── Images/
├── Anomaly_detection/                    # Research Paper 3: Anomaly Detection
│   ├── Anomaly_detection.tex
│   ├── Anomaly_detection.bib
│   ├── Results.ipynb                    # Jupyter notebook for plots
│   ├── generate_plots.py               # Python script for plot generation
│   ├── model_comparison_results.csv
│   └── clusterization_results.csv
└── ESSENTIAL_FILES.md                   # Documentation of essential files
```

## Quick Start

To build all documents (thesis, presentation, and research papers), simply run:

```bash
./build.sh
```

This script will:
1. Install required R packages
2. Generate all necessary data and plots
3. Compile all three research papers
4. Compile the main thesis (which includes the research papers)
5. Compile the presentation
6. Clean up auxiliary files

## Requirements

- **R** with packages: ggplot2, dplyr, gridExtra, metR, SpATS, gstat, sp, lme4, car, lmtest, spdep, scales
- **LaTeX** with pdflatex and bibtex
- **Python 3** (optional, for anomaly detection plots)

## Generated Output

After running the build script, you will have:

- `Thesis.pdf` - Complete PhD thesis document
- `Presentazione/PhD_Thesis_Presentation.pdf` - Thesis presentation
- `Plant_count/remotesensing-3622397.pdf` - Plant counting research paper
- `Phytotoxicity_score/Phytoxicity_score.pdf` - Phytotoxicity scoring research paper  
- `Anomaly_detection/Anomaly_detection.pdf` - Anomaly detection research paper

## Research Summary

This thesis investigates the application of geomatics technologies for recording spatially referenced observations in agricultural trials that comply with EPPO (European and Mediterranean Plant Protection Organization) standards. The research addresses three main variable types through dedicated studies:

1. **Plant Counting (Continuous/Discrete Variables)**: Using deep learning object detectors to count maize seedlings from UAV orthomosaics
2. **Phytotoxicity Scoring (Ordinal Variables)**: Using machine learning regressors with photogrammetric multispectral imaging
3. **Disease Detection (Nominal/Binary Variables)**: Using anomaly detection to classify healthy vs. diseased plant tissues

## Key Contributions

- First systematic evaluation of minimum requirements for implementing geomatics techniques within EPPO standards
- Demonstration that transformer-based object detectors require ~60 labeled images to achieve EPPO benchmark performance (R² > 0.85)
- Development of automated phytotoxicity scoring system achieving κ > 0.7 with only 30 training samples
- Proof that pre-trained models can achieve >85% accuracy in plant disease classification using anomaly detection

## License

This work is part of a PhD thesis submission. Please contact the author for usage permissions.

## Author

**Samuele Bumbaca**  
University of Turin  
samuele.bumbaca@unito.it
