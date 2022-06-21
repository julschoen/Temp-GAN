import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path): 
    self.files = np.load(path)['x']
    self.path = path[:-len(path.split('/')[-1])]
    self.len = len(self.files)

  def __getitem__(self, index):
      #pat = [os.path.join(self.path, self.files[index], f) for f in os.listdir(os.path.join(self.path, self.files[index])) if f.endswith('npz')][0]
      pat = os.path.join(self.path, self.files[index])
      x = np.load(pat)['x']
      ind = np.random.randint(0, min(x.shape[0]-2, 4))
      xs = x[ind:ind+3]
      
      xs_ = np.empty((3,64,128,128))
      try:
        for i, x in enumerate(xs):
          xs_[i] = np.flip(x.reshape(128,128,64).T,axis=0)
        xs = np.clip(xs_, -1,1)
      except:
        return self.__getitem__(index+1)
      return torch.from_numpy(xs).float().squeeze(), torch.Tensor([ind])

  def __len__(self):
      return self.len

class DataLIDC():
  def __init__(self, path):
    self.data = np.load(path)['x']
    self.path = path[:-len(path.split('/')[-1])]
    self.len = len(self.files)

  def __shift__(self, x, correct=True):
    if correct:
      ind = np.random.randint(0, min(x.shape[0]-2, 4))
      return x[ind:ind+3]
    else:
      i1 = np.random.randint(0, min(x.shape[0], 7))
      i2 = np.random.randint(0, min(x.shape[0], 7))
      i3 = np.random.randint(0, min(x.shape[0], 7))
      while s1 < s2 and s2 < s3:
        i1 = np.random.randint(0, min(x.shape[0], 7))
        i2 = np.random.randint(0, min(x.shape[0], 7))
        i3 = np.random.randint(0, min(x.shape[0], 7))

      x1 = x[i1]
      x1 = x[i2]
      x1 = x[i3]
      return np.concatenate((x1.reshape(1,128,128,-1),x2.reshape(1,128,128,-1),x3.reshape(1,128,128,-1)))

  def __getitem__(self, index):
    pat = os.path.join(self.path, self.files[index])
    image = np.load(pat)['x']
    if torch.rand(1)<0.51:
      image = self.__shift__(image)
      label = 1
    else:
      image = self.__shift__(image, correct=False)
      label = 0
    xs_ = np.empty((3,64,128,128))
    try:
      for i, x in enumerate(image):
        xs_[i] = np.flip(x.reshape(128,128,64).T,axis=0)
      image = np.clip(xs_, -1,1)
    except:
      return self.__getitem__(index+1)
    return torch.from_numpy(image).float(), torch.Tensor([label]).bool()

  def __len__(self):
    return self.len

