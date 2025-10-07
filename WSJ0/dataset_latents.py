# dataset_latents.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

def _to_tc(z) -> torch.Tensor:
    """
    Ensure [T,C]. Your cache script already saves [T,C],
    but this guards against accidental [C,T].
    """
    z = torch.as_tensor(z)
    if z.ndim != 2:
        raise ValueError(f"latent must be 2D [T,C], got {tuple(z.shape)}")
    if z.shape[0] < z.shape[1]:  # likely [C,T]
        z = z.transpose(0, 1).contiguous()
    return z

class RWKVLatentDataset(Dataset):
    """
    Expects a root like:
      root/
        mix/**/<utt>.pt
        s1/**/<utt>.pt
        s2/**/<utt>.pt

    Each .pt payload: {"z":[T,C], "fps":int, "sr":int, ...}
    """
    def __init__(
        self,
        root: str | Path,
        require_targets: bool = True,
        extensions: Tuple[str,...] = (".pt",),
    ):
        self.root = Path(root)
        self.mix_dir = self.root / "mix"
        self.s1_dir  = self.root / "s1"
        self.s2_dir  = self.root / "s2"
        self.require_targets = require_targets
        self.extensions = extensions

        assert self.mix_dir.is_dir(), f"Missing {self.mix_dir}"
        if require_targets:
            assert self.s1_dir.is_dir() and self.s2_dir.is_dir(), "Missing s1/ or s2/"

        # Build list of utterances by intersecting file stems
        mix_files = [p for p in self.mix_dir.rglob("*") if p.suffix in extensions]
        if require_targets:
            s1_set = {p.relative_to(self.s1_dir).with_suffix("").as_posix() for p in self.s1_dir.rglob("*") if p.suffix in extensions}
            s2_set = {p.relative_to(self.s2_dir).with_suffix("").as_posix() for p in self.s2_dir.rglob("*") if p.suffix in extensions}

        items = []
        for m in mix_files:
            rel = m.relative_to(self.mix_dir).with_suffix("")  # preserve subdirs
            if require_targets:
                key = rel.as_posix()
                if key not in s1_set or key not in s2_set:
                    continue
            items.append(rel)

        assert len(items) > 0, f"No items found under {self.root}"
        self.items = sorted(items)

    def __len__(self) -> int:
        return len(self.items)

    def _load_payload(self, p: Path) -> Dict:
        obj = torch.load(p, map_location="cpu")
        # Required keys
        assert "z" in obj, f"Missing 'z' in {p}"
        return obj

    def __getitem__(self, idx: int) -> Dict:
        rel = self.items[idx]
        mix_p = self.mix_dir / (str(rel) + ".pt")
        mix_obj = self._load_payload(mix_p)
        zmix = _to_tc(mix_obj["z"])

        ex: Dict = {
            "utt_id": str(rel),
            "z_mix": zmix,                         # [T,C]
            "fps": int(mix_obj.get("fps", 320)),
            "sr":  int(mix_obj.get("sr", 16000)),
        }

        if self.require_targets:
            s1_p = self.s1_dir / (str(rel) + ".pt")
            s2_p = self.s2_dir / (str(rel) + ".pt")
            s1_obj = self._load_payload(s1_p)
            s2_obj = self._load_payload(s2_p)
            ex["z_s1"] = _to_tc(s1_obj["z"])      # [T,C]
            ex["z_s2"] = _to_tc(s2_obj["z"])      # [T,C]

        return ex
