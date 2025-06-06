#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Convert the model if needed
python convert_model.py

# Create necessary directories
mkdir -p static/uploads
mkdir -p visualizations 