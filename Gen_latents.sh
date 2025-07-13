#!/bin/bash

chmod +x cache_latents.py

# Generate Latents
python cache_latents.py --csv train.csv --out_dir ~/thesis1/latents/train

python cache_latents.py --csv dev.csv --out_dir ~/thesis1/latents/dev

python cache_latents.py --csv test.csv --out_dir ~/thesis1/latents/test
