#!/bin/bash

# PhD Thesis Presentation Generator
# This script helps generate the presentation in different formats

echo "=== PhD Thesis Presentation Generator ==="
echo "Author: Samuele Bumbaca"
echo "University of Turin"
echo ""

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: Pandoc is not installed. Please install pandoc to generate presentations."
    echo "Ubuntu/Debian: sudo apt-get install pandoc"
    echo "macOS: brew install pandoc"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "PhD_Thesis_Presentation.md" ]; then
    echo "Error: PhD_Thesis_Presentation.md not found in current directory"
    echo "Please run this script from the Presentazione folder"
    exit 1
fi

echo "Available presentation formats:"
echo "1. PDF (Beamer LaTeX)"
echo "2. HTML (reveal.js)"
echo "3. PowerPoint (PPTX)"
echo "4. All formats"
echo ""

read -p "Select format (1-4): " choice

case $choice in
    1)
        echo "Generating PDF presentation..."
        pandoc PhD_Thesis_Presentation.md \
            --from markdown \
            --to beamer \
            --slide-level=2 \
            --output PhD_Thesis_Presentation.pdf \
            --metadata-file=presentation_config.yaml
        echo "Generated: PhD_Thesis_Presentation.pdf"
        ;;
    2)
        echo "Generating HTML presentation..."
        pandoc PhD_Thesis_Presentation.md \
            --from markdown \
            --to revealjs \
            --standalone \
            --slide-level=2 \
            --output PhD_Thesis_Presentation.html \
            --metadata-file=presentation_config.yaml \
            --variable theme=white \
            --variable transition=slide
        echo "Generated: PhD_Thesis_Presentation.html"
        ;;
    3)
        echo "Generating PowerPoint presentation..."
        pandoc PhD_Thesis_Presentation.md \
            --from markdown \
            --to pptx \
            --slide-level=2 \
            --output PhD_Thesis_Presentation.pptx \
            --metadata-file=presentation_config.yaml
        echo "Generated: PhD_Thesis_Presentation.pptx"
        ;;
    4)
        echo "Generating all formats..."
        
        # PDF
        pandoc PhD_Thesis_Presentation.md \
            --from markdown \
            --to beamer \
            --slide-level=2 \
            --output PhD_Thesis_Presentation.pdf \
            --metadata-file=presentation_config.yaml
        
        # HTML
        pandoc PhD_Thesis_Presentation.md \
            --from markdown \
            --to revealjs \
            --standalone \
            --slide-level=2 \
            --output PhD_Thesis_Presentation.html \
            --metadata-file=presentation_config.yaml \
            --variable theme=white \
            --variable transition=slide
        
        # PowerPoint
        pandoc PhD_Thesis_Presentation.md \
            --from markdown \
            --to pptx \
            --slide-level=2 \
            --output PhD_Thesis_Presentation.pptx \
            --metadata-file=presentation_config.yaml
        
        echo "Generated all formats:"
        echo "- PhD_Thesis_Presentation.pdf"
        echo "- PhD_Thesis_Presentation.html"
        echo "- PhD_Thesis_Presentation.pptx"
        ;;
    *)
        echo "Invalid choice. Please select 1-4."
        exit 1
        ;;
esac

echo ""
echo "Presentation generation complete!"
echo ""
echo "Presentation timing (40 minutes total):"
echo "- Introduction & Background: 20 minutes (Slides 1-20)"
echo "- Study 1 (Plant Counting): 6 minutes (Slides 21-27)"
echo "- Study 2 (Phytotoxicity): 6 minutes (Slides 28-34)"
echo "- Study 3 (Anomaly Detection): 6 minutes (Slides 35-41)"
echo "- Conclusions: 2 minutes (Slides 42-46)"
echo ""
echo "Good luck with your defense!"
