import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path): 
    self.path = path
    self.files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
    self.len = len(self.files)

  def __getitem__(self, index):
      x = np.load(self.files[0])['x']
      ind = np.random.choice(x.shape[0], 3)
      xs = x[ind]
      xs = np.clip(xs, -1,1)
      return torch.from_numpy(xs).float()

  def __len__(self):
      return self.len

