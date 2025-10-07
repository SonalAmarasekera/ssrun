import torch, dac
#from models.separator_rwkv6 import RWKV6Separator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

codec = dac.DAC.load(dac.utils.download("16khz")).to(device).eval()
sep   = RWKV6Separator(codec).to(device)

wav   = torch.randn(1, 1, 64000, device=device)   # dummy mixture
pred, _ = sep(wav)                                # ✅ no dtype mismatch
assert pred[0].shape[1] == 1024
print("Forward OK on", device)
