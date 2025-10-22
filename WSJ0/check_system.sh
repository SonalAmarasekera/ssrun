#!/bin/bash

# GPU
nvidia-smi

#torch version
python -c "import torch; print('torch', torch.version, 'cuda', torch.version.cuda)"

#cuda toolkit version
nvcc --version
