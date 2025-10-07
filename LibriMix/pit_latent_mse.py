#Loss: losses/pit_latent_mse.py

import torch
import torch.nn.functional as F
def pit_mse(preds, tgt1, tgt2):

    loss_a = F.mse_loss(preds[0], tgt1) + F.mse_loss(preds[1], tgt2)
    loss_b = F.mse_loss(preds[0], tgt2) + F.mse_loss(preds[1], tgt1)
    return torch.min(loss_a, loss_b)
