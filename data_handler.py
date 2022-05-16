import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DATA(Dataset):
  def __init__(self, path): 
    self.data = np.load(path)['X']
    self.len = self.data.shape[0]

  def __getitem__(self, index):
      image = self.data[index]
      image = np.clip(image, -1,1)
      return torch.from_numpy(image).float()

  def __len__(self):
      return self.len

