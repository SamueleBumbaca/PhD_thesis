# PhD Thesis Presentation - Slide Deck
## 40-Minute Defense Presentation

This presentation covers the PhD thesis "Geomatics Technologies for Enhanced Plant Protection Product Efficacy Evaluation: Integrating Spatial Data with Geostatistical Methods" by Samuele Bumbaca.

## Presentation Structure

### Part 1: Introduction & Background (20 minutes - Slides 1-20)
- Research problem and motivation
- EPPO framework and regulatory context
- Theoretical background (geostatistics, geomatics, ML)
- Research objectives and methodology

### Part 2: Three Case Studies (18 minutes - Slides 21-41)
#### Study 1: Plant Counting (6 minutes - Slides 21-27)
- Continuous/discrete variables
- Object detection methodology
- Dataset requirements findings

#### Study 2: Phytotoxicity Scoring (6 minutes - Slides 28-34)
- Ordinal variables
- Multispectral photogrammetry
- ML regression approach

#### Study 3: Anomaly Detection (6 minutes - Slides 35-41)
- Binary/nominal variables
- Pre-trained model evaluation
- Unsupervised classification

### Part 3: Conclusions (2 minutes - Slides 42-46)
- Overall achievements
- Future directions
- Impact and implementation

## Key Messages

1. **Problem**: Traditional PPP trials rely on human judgment for environmental variability assessment
2. **Solution**: Geomatics technologies provide automatic spatial coordinates enabling geostatistical analysis
3. **Validation**: All three EPPO variable types successfully addressed with benchmark performance
4. **Impact**: Transformative potential for agricultural research and regulatory evaluation

## Technical Achievements

- **Plant Counting**: R² > 0.85 with 60-130 training images (architecture-dependent)
- **Phytotoxicity**: κ > 0.7 with only 30 training samples, ordinal-to-continuous conversion
- **Disease Detection**: >85% accuracy using pre-trained models without task-specific training

## Regulatory Compliance

All methods meet or exceed EPPO standard requirements:
- R² > 0.85 benchmark achieved
- Spatial coordinates captured for geostatistical analysis
- Objective, reproducible measurements
