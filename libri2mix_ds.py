#------------------------------Dataset Class--------------------------------------
# Should move to datasets.libri2mix_ds
import torch
import pandas
import soundfile as sf
import torch.nn.functional as F
import pathlib

class Libri2MixDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, segment=4.0, sr=16000, cache_latents=False, lat_root=None, num_workers=0):
        self.df = pandas.read_csv(csv_path)
        self.seg_len = int(segment * sr)
        self.sr = sr
        self.cache_latents = cache_latents
        # 1️⃣  Determine where latent .pt files live
        if cache_latents:
            if lat_root is not None:
                self.lat_dir = pathlib.Path(lat_root).expanduser().resolve()
            else:
                # default: replace the split folder name with 'latents/<split>'
                split_name = pathlib.Path(csv_path).stem.split('.')[0]  # train / dev / test
                self.lat_dir = pathlib.Path("latents") / split_name
            assert self.lat_dir.exists(), f"latent dir {self.lat_dir} missing"
        else:
            self.lat_dir = None

    def __len__(self): return len(self.df)

    def _load_wav(self, path):
        wav, sr = sf.read(path, dtype='float32')
        assert sr == self.sr
        return torch.from_numpy(wav)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mix = self._load_wav(row.mix_path)
        s1  = self._load_wav(row.s1_path)
        s2  = self._load_wav(row.s2_path)

        # 1-channel mono checks
        if mix.ndim==2: mix = mix.mean(1)
        # Random 4-s crop
        if mix.size(0) > self.seg_len:
            start = torch.randint(0, mix.size(0)-self.seg_len, (1,)).item()
            self.last_start = start
            mix  = mix[start:start+self.seg_len]
            s1   = s1[start:start+self.seg_len]
            s2   = s2[start:start+self.seg_len]
        else:
            self.last_start = 0
            pad = self.seg_len - mix.size(0)
            mix = F.pad(mix, (0,pad))
            s1  = F.pad(s1 , (0,pad))
            s2  = F.pad(s2 , (0,pad))

        mix  = mix.unsqueeze(0)          # [1,T]
        srcs = torch.stack([s1,s2],0)    # [2,T]

        stem = pathlib.Path(row.mix_path).stem
        lat_tuple = (
            str(self.lat_dir / f"{stem}_s1.pt"),
            str(self.lat_dir / f"{stem}_s2.pt"),
)

        if self.cache_latents:
            srcs = None                   # to save RAM
        return mix, srcs, lat_tuple, self.last_start
