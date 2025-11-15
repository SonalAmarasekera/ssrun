#!/bin/bash
set -e

# WSJ0 root data download and move
echo "Downloading dataset root folders..."
pip install kagglehub
python wsj0mix_kagglehub.py

# Cloning the needed repos
echo "Cloning necessary repos..."
git clone https://github.com/mpariente/pywsj0-mix.git
git clone https://github.com/BlinkDL/RWKV-LM.git
git clone https://github.com/descriptinc/descript-audio-codec.git

# Required packages
echo "Installing required packages..."
pip install ninja wandb pandas scipy numpy tensorboard seaborn
pip install git+https://github.com/descriptinc/descript-audio-codec

#Rename DAC
mv descript-audio-codec DAC

#Moving to RWKV directory and installing dependencies
cd RWKV-LM/RWKV-v7/train_temp
rm -f src/model.py
pip install -r requirements.txt
cd ../../../

# Renaming RWKV repo for access inside script
mv RWKV-LM/RWKV-v7 RWKV-LM/RWKV_v7
mv RWKV-LM RWKV

# Moving the edited model.py inside the original repo and cuda folder to the main area
mv model.py RWKV/RWKV_v7/train_temp/src/
cp -r RWKV/RWKV_v7/train_temp/cuda .

# Creating the WSJ0mix
cd pywsj0-mix/

echo "Starting data creation..."

# Generate data for 2 speakers at 8k
python generate_wsjmix.py -p /root/.cache/kagglehub/datasets/sonishmaharjan555/wsj0-2mix/versions/2/ -o /workspace/ -n 2 -sr 16000 --len_mode min

cd ../

# Generating latents
echo "Generating latents (.dac)..."
python fixed_dac_encoder.py /content/2speakers/wav16k/min/ --output /content/latents/ --model_type "16khz" --batch_size 10 --device "cuda"

# Renaming for easier access
mv 

# Check files for similarity
echo "Checking similarity..."
python test_latent_differences.py --latent_dir /workspace/latents/min/train --quick --num_samples 20000
python test_latent_differences.py --latent_dir /workspace/latents/min/dev --quick --num_samples 5000
python test_latent_differences.py --latent_dir /workspace/latents/min/test --quick --num_samples 3000
