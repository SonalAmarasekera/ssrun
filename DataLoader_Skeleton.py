#---------DataLoader Skeleton------------
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=1,
                    shuffle=True,
                    num_workers=2, # Can't use 4 as was suggested
                    pin_memory=True)
