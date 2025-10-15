#!/bin/bash

chmod +x cache_latents_working.py

# Generate Latents
#python cache_latents_working.py --csv train_min.csv --out_dir /content/latents/min/train --model_type 16khz --device cuda

python cache_latents_working.py --csv dev_min.csv --out_dir /content/latents/min/dev --model_type 16khz --device cuda

python cache_latents_working.py --csv test_min.csv --out_dir /content/latents/min/test --model_type 16khz --device cuda
