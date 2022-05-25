import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path): 
    self.files = np.load(path)['x']
    self.path = path.split('/')[0]
    self.len = len(self.files)

  def __getitem__(self, index):
      pat = [os.path.join(self.path, self.files[index], f) for f in os.listdir(os.path.join(self.path, self.files[index])) if f.endswith('npz')][0]
      x = np.load(pat)['x']
      ind = np.random.choice(x.shape[0], 3)
      xs = x[ind]
      xs_ = np.empty((3,64,128,128))
      for i, x in enumerate(xs):
        xs_[i] = np.flip(x.reshape(128,128,64).T,axis=0)
      xs = np.clip(xs_, -1,1)
      return torch.from_numpy(xs).float().squeeze(), torch.Tensor(ind)

  def __len__(self):
      return self.len

