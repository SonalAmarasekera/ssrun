#!/bin/bash

# Cloning the needed repos
git clone https://github.com/mpariente/pywsj0-mix.git

# Install kagglehub
pip install kagglehub

# Get the root files from Kagglehub
python wsj0mix_kagglehub.py

# Move the data files to accessible space
mv /root/.cache/kagglehub/datasets/sonishmaharjan555/wsj0-2mix/versions/2 /content/

cd pywsj0-mix/

echo "Starting data creation..."
# Generate data for 2 speakers at 8k
python generate_wsjmix.py -p /content/2 -o /content/ -n 2 -sr 8000
