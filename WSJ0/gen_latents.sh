#!/bin/bash

chmod +x cache_latents_rwkv_v7_fixed_pb.py

# Generate Latents
python cache_latents_rwkv_v7_fixed_pb.py --csv train_new.csv --out_dir /content/latents/train

python cache_latents_rwkv_v7_fixed_pb.py --csv dev_new.csv --out_dir /content/latents/dev

python cache_latents_rwkv_v7_fixed_pb.py --csv test_new.csv --out_dir /content/latents/test
