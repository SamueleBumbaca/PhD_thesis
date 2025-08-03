# Alternative way to connect R to VS Code
# Run this in your R console if .vsc.attach() doesn't work

# Load the languageserver package
library(languageserver)

# Start the language server
languageserver::run()
