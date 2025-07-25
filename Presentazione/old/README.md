# PhD Thesis Presentation Materials

This folder contains all materials needed for the 40-minute PhD thesis defense presentation on "Geomatics Technologies for Enhanced Plant Protection Product Efficacy Evaluation: Integrating Spatial Data with Geostatistical Methods" by Samuele Bumbaca.

## Files Overview

### Core Presentation Files
- **`PhD_Thesis_Presentation.md`** - Main presentation in Markdown format (46 slides)
- **`PhD_Thesis_Presentation.html`** - Interactive HTML version with navigation
- **`Speaker_Notes.md`** - Detailed talking points and timing guidance
- **`Presentation_Overview.md`** - Summary and key messages

### Configuration & Generation
- **`presentation_config.yaml`** - Pandoc configuration for format conversion
- **`generate_presentation.sh`** - Script to generate PDF, HTML, and PowerPoint versions
- **`README.md`** - This file

### Assets
- **`Imgs/`** - Folder for presentation images and graphics
  - `Logo_unito.webp` - University of Turin logo

## Presentation Structure (40 minutes)

### Part 1: Introduction & Background (20 minutes)
- **Slides 1-20**: Research problem, EPPO framework, theoretical background
- **Key focus**: Establishing the research gap and proposed solution

### Part 2: Three Case Studies (18 minutes)
- **Slides 21-27**: Study 1 - Plant Counting (6 minutes)
- **Slides 28-34**: Study 2 - Phytotoxicity Scoring (6 minutes)  
- **Slides 35-41**: Study 3 - Anomaly Detection (6 minutes)

### Part 3: Conclusions (2 minutes)
- **Slides 42-46**: Overall achievements, future work, and impact

## How to Use These Materials

### 1. Quick Start - HTML Version
- Open `PhD_Thesis_Presentation.html` in any web browser
- Use arrow keys or space bar to navigate
- Click navigation menu for quick jumping between sections
- Works offline, no internet required

### 2. Generate Multiple Formats
```bash
# Make script executable (first time only)
chmod +x generate_presentation.sh

# Run the generator
./generate_presentation.sh

# Choose format:
# 1 - PDF (Beamer LaTeX)
# 2 - HTML (reveal.js) 
# 3 - PowerPoint (PPTX)
# 4 - All formats
```

### 3. Manual Generation with Pandoc
```bash
# PDF version
pandoc PhD_Thesis_Presentation.md -t beamer -o presentation.pdf --slide-level=2

# HTML version (reveal.js)
pandoc PhD_Thesis_Presentation.md -t revealjs -s -o presentation.html

# PowerPoint version
pandoc PhD_Thesis_Presentation.md -t pptx -o presentation.pptx
```

## Presentation Tips

### Timing Guidelines
- **Average per slide**: 30-60 seconds
- **Introduction slides**: ~30 seconds each
- **Technical slides**: ~45 seconds each
- **Results slides**: ~60 seconds each
- **Conclusion slides**: ~30 seconds each

### Key Messages to Emphasize
1. **Problem significance**: Current agricultural statistics limitations
2. **Technical innovation**: First systematic geomatics evaluation in EPPO framework
3. **Practical results**: All methods exceed regulatory benchmarks
4. **Ready for adoption**: Clear implementation guidelines provided

### Presentation Flow
1. **Start strong**: Clear problem statement
2. **Build systematically**: Each study addresses one variable type
3. **Show results clearly**: Emphasize benchmark achievements
4. **End with impact**: Practical implications for agriculture

## Technical Requirements

### For HTML Version
- Any modern web browser
- No additional software needed

### For PDF/PowerPoint Generation
- **Pandoc** (document converter)
  - Ubuntu/Debian: `sudo apt-get install pandoc`
  - macOS: `brew install pandoc`
  - Windows: Download from https://pandoc.org/installing.html
- **LaTeX** (for PDF generation)
  - Ubuntu/Debian: `sudo apt-get install texlive-full`
  - macOS: Install MacTeX
  - Windows: Install MiKTeX

## Presentation Equipment Setup

### Recommended Setup
- **Laptop with backup**: Primary and backup presentation devices
- **HDMI/VGA adapters**: For projector connection
- **Wireless remote**: For slide advancement
- **Backup formats**: Have PDF, HTML, and PowerPoint versions ready

### Backup Plans
- **USB with all formats**: In case laptop fails
- **Printed slides**: Emergency hard copies
- **Online backup**: Cloud storage access

## Question Preparation

### Likely Questions & Preparation
1. **Computational requirements**: Discuss edge computing solutions
2. **Scalability concerns**: Address cloud processing options
3. **Regulatory acceptance**: Emphasize EPPO benchmark compliance
4. **Cost-benefit analysis**: Highlight efficiency improvements
5. **Generalization**: Discuss applicability to other crops

### Additional Materials
- **Technical appendix**: Detailed methodology available
- **Code repositories**: Links to implementation
- **Data samples**: Example datasets for demonstration

## Success Metrics

### Presentation Goals
- **Clear communication**: Complex technical concepts made accessible
- **Regulatory relevance**: Practical application emphasized
- **Scientific rigor**: Proper validation and benchmarking
- **Future impact**: Transformative potential demonstrated

### Defense Objectives
- **Demonstrate expertise**: Comprehensive understanding of field
- **Show innovation**: Novel contributions to agriculture and geomatics
- **Prove readiness**: Prepared for independent research career
- **Create impact**: Work ready for immediate application

## Contact Information

**Presenter**: Samuele Bumbaca  
**Institution**: University of Turin  
**Email**: samuele.bumbaca@unito.it  
**Defense Date**: July 17, 2025

## Publications Referenced

1. "On the Minimum Dataset Requirements for Fine-Tuning an Object Detector for Arable Crop Plant Counting: A Case Study on Maize Seedlings" - *Remote Sensing*, DOI: 10.3390/rs17132190
2. "Supporting Screening of New Plant Protection Products through a Multispectral Photogrammetric Approach Integrated with AI" - *Agronomy*, DOI: 10.3390/agronomy14020306
3. "Anomaly Detection for Plant Disease Classification" - *In preparation for submission*

---

**Good luck with your defense! The materials are comprehensive and ready for professional presentation.**
