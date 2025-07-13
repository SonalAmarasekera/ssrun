#!/bin/bash

# Cloning the needed repos
git clone https://github.com/JorisCos/LibriMix.git
git clone https://github.com/descriptinc/descript-audio-codec.git
git clone https://github.com/RWKV/RWKV-block.git

# Install requirements per repo
pip install -r LibriMix/requirements.txt
pip install descript-audio-codec
pip install flash-linear-attention
pip install "mir_eval>=0.7"

# Moving around files
mv generate_librimix_clean16k_2spk.sh LibriMix/
mv create_librimix_from_metadata_new.py LibriMix/scripts
mkdir LibriMix/extra
mv LibriMix/metadata/Libri2Mix/libri2mix_train-clean-360.csv LibriMix/extra/
mv LibriMix/metadata/Libri2Mix/libri2mix_train-clean-360_info.csv LibriMix/extra/
mv make_csv.py LibriMix/

cd LibriMix

# Downloading flac files of LibriMix
chmod +x generate_librimix_clean16k_2spk.sh
mkdir dataset
./generate_librimix_clean16k_2spk.sh  dataset/

# Making the mapping csv
chmod +x make_csv.py

# TRAIN_SET (100 h subset)
python make_csv.py --root "dataset/Libri2Mix/wav16k/min/train-100" --out ~/thesis1/train.csv
# DEV_SET
python make_csv.py --root "dataset/Libri2Mix/wav16k/min/dev" --out ~/thesis1/dev.csv
# TEST_SET
python make_csv.py --root "dataset/Libri2Mix/wav16k/min/test" --out ~/thesis1/test.csv
