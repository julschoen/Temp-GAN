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
    self.data = np.load(path)['X']
    self.len = self.data.shape[0]

  def __shift__(self, x):
    s1 = np.random.randint(0,40,1)[0]
    s2 = np.random.randint(0,20,1)[0]
    s3 = np.random.randint(0,20,1)[0]
    x1 = np.pad(x, [[0,0],[0, 0],[s1,0]], constant_values=-1)[:,:,:128]
    x2 = np.pad(x1, [[0,0],[0, 0],[s2,0]], constant_values=-1)[:,:,:128]
    x3 = np.pad(x2, [[0,0],[0, 0],[s3,0]], constant_values=-1)[:,:,:128]
    return nnp.concatenate((x1.reshape(1,128,128,128),x2.reshape(1,128,128,128),x3.reshape(1,128,128,128)))

  def __getitem__(self, index):
    image = self.data[index]
    image = np.clip(image, -1,1)
    image = self.__shift__(image)
    return torch.from_numpy(image).float(), torch.Tensor([0])

  def __len__(self):
    return self.len

