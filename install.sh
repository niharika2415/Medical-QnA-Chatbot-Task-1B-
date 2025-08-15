#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install system-level dependencies.
# The `pip install` command can fail if a Python package needs a system
# package to compile from source. We can install these first.
# Here, `g++` and other tools are needed for `blis`.
# We'll also install the `spacy` model as a pre-compiled binary here.
# Note: apt-get is used for Debian-based systems like the one on Streamlit Cloud.
sudo apt-get update
sudo apt-get install -y build-essential

# This line installs the pre-compiled `blis` wheel for Python 3.13.
# The URL might need to be updated if the Python version on Streamlit changes.
# You can find the correct URL on the spacy-models GitHub releases page.
pip install https://github.com/explosion/spacy-models/releases/download/en_core_sci_sm-3.4.4/en_core_sci_sm-3.4.4.tar.gz

# Now install the remaining Python dependencies from requirements.txt
pip install -r requirements.txt

# All done!
echo "Installation complete."
