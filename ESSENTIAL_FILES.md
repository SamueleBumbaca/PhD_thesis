# Essential Files for PhD Thesis Repository

## Root Directory Files (Keep)
- Thesis.tex (main thesis document)
- Phd_Thesis_SBumbaca.bib (bibliography)
- build.sh (comprehensive build script)

## Essential R Scripts (Keep)
- Presentazione/install_packages.R
- Presentazione/generate_trial_data.R
- Presentazione/integrated_trial_analysis.R
- Presentazione/integrated_trial_analysis_irregular.R

## Presentation Files (Keep)
- Presentazione/PhD_Thesis_Presentation.tex
- Presentazione/Imgs/ (entire folder - contains all images)

## Research Papers (Keep)
- Plant_count/remotesensing-3622397.tex (Plant counting research paper)
- Plant_count/remotesensing-3622397.bib (bibliography)
- Plant_count/Unsupervised_plant_counting.bib (additional bibliography)
- Phytotoxicity_score/Phytoxicity_score.tex (Phytotoxicity scoring research paper)
- Phytotoxicity_score/SAGEA_DiSAFA.bib (bibliography)
- Phytotoxicity_score/Images/ (folder with paper images)
- Anomaly_detection/Anomaly_detection.tex (Anomaly detection research paper)
- Anomaly_detection/Anomaly_detection.bib (bibliography)
- Anomaly_detection/Results.ipynb (Jupyter notebook for plots)
- Anomaly_detection/generate_plots.py (Python script for plot generation)

## Required PDFs for Inclusion (Keep)
- Intestazione/t3._Thesis_first_page.pdf
- Plant_count/remotesensing-3622397.pdf (will be generated)
- Phytotoxicity_score/Phytoxicity_score.pdf (will be generated)
- Anomaly_detection/Anomaly_detection.pdf (will be generated)

## Generated Data Files (Keep if exist, will be regenerated)
- dataset1_detailed.csv
- dataset2_plots.csv
- trial_models.RData
- integrated_rcbd_spats_comparison.png
- integrated_rcbd_spats_comparison_irregular.png

## Generated Plot Files for Research Papers (Keep if exist, will be regenerated)
- Anomaly_detection/Plant_Pathology_Dataset_Anomaly_Detection_Performance.pdf
- Anomaly_detection/Plant_Village_Dataset_Anomaly_Detection_Performance.pdf
- Anomaly_detection/Plant_Pathology_Dataset_Clustering_Performance.pdf
- Anomaly_detection/Plant_Village_Dataset_Clustering_Performance.pdf
- Anomaly_detection/model_comparison_results.csv
- Anomaly_detection/clusterization_results.csv

## Files to Remove
- All .aux, .log, .out, .toc, .bbl, .blg, .synctex.gz files
- All test_*.R files in Presentazione/
- Presentazione/old/ folder
- All .bat files
- All markdown files (.md) except this one
- All backup files
- Unused R scripts
- texput.log files
- Anomaly_detection/Results copy.ipynb (duplicate notebook)

## Build Process
1. Run R scripts to generate trial data and plots
2. Run Python scripts to generate anomaly detection plots (if available)
3. Compile all research papers (Plant_count, Phytotoxicity_score, Anomaly_detection)
4. Compile main thesis (includes research papers as PDFs)
5. Compile presentation
6. Clean up auxiliary files
