#!/bin/bash

# Cloning the needed repos
git clone https://github.com/mpariente/pywsj0-mix.git

# Install kagglehub
pip install kagglehub
pip install ninja
pip install torch pytorch-lightning==1.9.5 deepspeed

# Get the root files from Kagglehub
python wsj0mix_kagglehub.py

# Move the data files to accessible space
mv /root/.cache/kagglehub/datasets/sonishmaharjan555/wsj0-2mix/versions/2 /content/

cd pywsj0-mix/

echo "Starting data creation..."
# Generate data for 2 speakers at 8k
python generate_wsjmix.py -p /content/2 -o /content/ -n 2 -sr 8000

# Mapping the CSVs
python make_csv.py --root /content/2speakers/wav8k/min/tr --out train_min.csv
python make_csv.py --root /content/2speakers/wav8k/min/cv --out dev_min.csv
python make_csv.py --root /content/2speakers/wav8k/min/tt --out test_min.csv

python make_csv.py --root /content/2speakers/wav8k/max/tr --out train_max.csv
python make_csv.py --root /content/2speakers/wav8k/max/cv --out dev_max.csv
python make_csv.py --root /content/2speakers/wav8k/max/tt --out test_max.csv
