#!/bin/bash

# Compile LaTeX Beamer presentation
echo "Compiling LaTeX Beamer presentation..."

# First compilation
pdflatex PhD_Thesis_Presentation.tex

# Second compilation (for references)
pdflatex PhD_Thesis_Presentation.tex

# Clean auxiliary files
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

echo "Compilation complete! PDF generated: PhD_Thesis_Presentation.pdf"

# Optional: Open the PDF
if command -v xdg-open &> /dev/null; then
    xdg-open PhD_Thesis_Presentation.pdf
elif command -v open &> /dev/null; then
    open PhD_Thesis_Presentation.pdf
fi