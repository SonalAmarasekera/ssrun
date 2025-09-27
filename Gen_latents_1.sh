#!/bin/bash

chmod +x cache_latents_1.py

# Generate Latents
python3 cache_latents_1.py --csv train.csv --out_dir ~/thesis1/latents/train-100

python3 cache_latents_1.py --csv dev.csv --out_dir ~/thesis1/latents/dev

python3 cache_latents_1.py --csv test.csv --out_dir ~/thesis1/latents/test