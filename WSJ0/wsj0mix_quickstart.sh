#!/bin/bash

# Cloning the needed repos
git clone https://github.com/mpariente/pywsj0-mix.git
git clone https://github.com/BlinkDL/RWKV-LM.git

# Install kagglehub
pip install kagglehub
pip install ninja wandb

# Get the root files from Kagglehub
python wsj0mix_kagglehub.py

#Moving to RWKV directory and installing dependencies
cd RWKV-LM/RWKV-v7/train_temp
pip install -r requirements.txt
cd ../../../

# Move the data files to accessible space
mv /root/.cache/kagglehub/datasets/sonishmaharjan555/wsj0-2mix/versions/2 /content/

cd pywsj0-mix/

echo "Starting data creation..."
# Generate data for 2 speakers at 8k
python generate_wsjmix.py -p /content/2 -o /content/ -n 2 -sr 8000

cd ../

# Mapping the CSVs
python make_csv.py --root /content/2speakers/wav8k/min/tr --out train_min.csv
python make_csv.py --root /content/2speakers/wav8k/min/cv --out dev_min.csv
python make_csv.py --root /content/2speakers/wav8k/min/tt --out test_min.csv

python make_csv.py --root /content/2speakers/wav8k/max/tr --out train_max.csv
python make_csv.py --root /content/2speakers/wav8k/max/cv --out dev_max.csv
python make_csv.py --root /content/2speakers/wav8k/max/tt --out test_max.csv
