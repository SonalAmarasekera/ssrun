import pathlib
from datasets.libri2mix_ds import Libri2MixDataset

ds = Libri2MixDataset("~/train.csv", cache_latents=True)
mix, srcs, lat_paths = ds[0]
assert mix.shape == (1, 64000)
if srcs == None:
    pass
else:
    assert srcs.shape == (2, 64000)
assert len(lat_paths) == 2
for p in lat_paths: pathlib.Path(p).exists()
print("Dataset returns 3-tuple ✔ and lat_dir =", ds.lat_dir)
