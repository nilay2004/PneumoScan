#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/uploads
mkdir -p visualizations
mkdir -p models

# Set environment variables
export PYTHONUNBUFFERED=1
export FLASK_APP=app.py
export FLASK_ENV=production

# Convert the model if needed
python convert_model.py 