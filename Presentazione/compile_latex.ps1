# Compile LaTeX Beamer presentation
Write-Host "Compiling LaTeX Beamer presentation..."

# First compilation
pdflatex PhD_Thesis_Presentation_clean.tex

# Second compilation (for references)
pdflatex PhD_Thesis_Presentation_clean.tex

# Clean auxiliary files
Remove-Item *.aux, *.log, *.nav, *.out, *.snm, *.toc, *.vrb -ErrorAction SilentlyContinue

Write-Host "Compilation complete! PDF generated: PhD_Thesis_Presentation_clean.pdf"

# Optional: Open the PDF
$PdfPath = "PhD_Thesis_Presentation_clean.pdf"
if (Test-Path $PdfPath) {
    Start-Process $PdfPath
}