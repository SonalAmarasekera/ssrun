#!/bin/bash

chmod +x cache_latents_working.py

# Generate Latents
python cache_latents_working.py --csv train.csv --out_dir /content/latents/train

python cache_latents_working.py --csv dev.csv --out_dir content/latents/dev

python cache_latents_working.py --csv test.csv --out_dir /content/latents/test
