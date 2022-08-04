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

class Data4D():
  def __init__(self, path, shift=True):
    self.files = np.load(path)['x']
    self.path = path[:-len(path.split('/')[-1])]
    self.len = len(self.files)
    self.shift = shift

  def __shift__(self, x, correct=True):
    if correct:
      ind = np.sort(np.random.choice([0,1,2,3,4], 3, replace=False))
      return x[ind]
    else:
      i1 = np.random.randint(0, 5)
      i2 = np.random.randint(0, 5)
      i3 = np.random.randint(0, 5)
      while i1 < i2 and i2 < i3:
        i1 = np.random.randint(0, 5)
        i2 = np.random.randint(0, 5)
        i3 = np.random.randint(0, 5)

      x1 = x[i1]
      x2 = x[i2]
      x3 = x[i3]
      return np.concatenate((x1.reshape(1,128,128,-1),x2.reshape(1,128,128,-1),x3.reshape(1,128,128,-1)))

  def __dif_pat__(self, x, index):
    ind = np.random.choice(range(self.len))
    while ind == index:
      ind = np.random.choice(range(self.len))

    pat = os.path.join(self.path, self.files[ind])
    x_ = np.load(pat)['x']

    ind = np.random.randint(0, 3)

    x_true = x[ind:ind+3]
    x_false = x_[ind:ind+3]
    ind = np.random.randint(0,3)
    x_true[ind] = x_false[ind]

    return x_true


  def __getitem__(self, index):
    if self.shift:
      pat = os.path.join(self.path, self.files[index])
      image = np.load(pat)['x']
      if torch.rand(1)<0.51:
        image = self.__shift__(image)
        label = 1
      else:
        if torch.rand(1)<0.51:
          image = self.__shift__(image, correct=False)
        else:
          image = self.__dif_pat__(image, index)
        label = 0
      xs_ = np.empty((3,64,128,128))
      for i, x in enumerate(image):
        xs_[i] = np.flip(x.reshape(128,128,64).T,axis=0)
      image = np.clip(xs_, -1,1)
      return torch.from_numpy(image).float(), torch.Tensor([label])
    else:
      pat = os.path.join(self.path, self.files[index])
      image = np.load(pat)['x']
      ind = np.random.randint(0, x.shape[0])
      image = np.flip(image[ind].reshape(128,128,64).T,axis=0)
      image = np.clip(xs_, -1,1)
      return torch.from_numpy(image).float()

  def __len__(self):
      return self.len


class DataLIDC():
  def __init__(self, path, triplet=False, shift=True):
    self.data = np.load(path)['X']
    self.len = 600#self.data.shape[0]
    self.shift = shift
    self.triplet = triplet
    self.shift_amount = np.arange(5,32)

  def __pad__(self, x, s):
    if s == 0:
      return x.copy()
    elif s > 0:
        return np.pad(x.copy(), [[0,0],[0, 0],[s,0]], constant_values=-1)[:,:,:128]
    else:
        return np.pad(x.copy(), [[0,0],[0, 0],[0,np.abs(s)]], constant_values=-1)[:,:,np.abs(s):]

  def __get_shift__(self, sort=True):
    if sort:
      s1, s2, s3 = np.sort(np.random.choice(self.shift_amount,3, replace=False))
      p = torch.rand(1)
      if p < 0.25:
        s1, s2, s3 = -s3, -s2, -s1
      elif p < 0.50:
        s1, s2 = -s2, -s1
      elif p < 0.75:
        s1, s2 = -s2, s1
      return s1,s2,s3
    else:
      s1, s2, s3 = np.random.choice(np.arange(-32,32),3, replace=False)
      return s1,s2,s3


  def __shift__(self, x, correct=True):
    if correct:
      s1, s2, s3 = self.__get_shift__()
      x1 = self.__pad__(x, s1)
      x2 = self.__pad__(x, s2)
      x3 = self.__pad__(x, s3)
    else:
      s1, s2, s3 = self.__get_shift__(sort=False)
      while s1 < s2 and s2 < s3:
        s1, s2, s3 = self.__get_shift__(sort=False)
      x1 = self.__pad__(x, s1)
      x2 = self.__pad__(x, s2)
      x3 = self.__pad__(x, s3)
  
    return np.concatenate((x1.reshape(1,128,128,-1),x2.reshape(1,128,128,-1),x3.reshape(1,128,128,-1)))

  def __dif_pat__(self, x, index):
    ind = np.random.choice(range(self.len))
    while ind == index:
      ind = np.random.choice(range(self.len))

    x_ = self.data[ind]
    s1, s2 = self.__get_shift__()
    if s1<0:
      if s2 < 0:
        x1 = self.__pad__(x, s1)
        x2 = self.__pad__(x, s2)
        x3 = x.copy()
      else:
        x1 = self.__pad__(x, s1)
        x2 = x.copy()
        x3 = self.__pad__(x, s2)
    else:
      x1 = x.copy()
      x2 = self.__pad__(x, s1)
      x3 = self.__pad__(x, s2)

    p = torch.rand(1)
    if p < 0.33:
      x1 = x_
    elif p < 0.66:
      x2 = x_
    else:
      x3 = x_

    return np.concatenate((x1.reshape(1,128,128,-1),x2.reshape(1,128,128,-1),x3.reshape(1,128,128,-1)))

  def __getitem__(self, index):
    if self.triplet and self.shift:
      image = self.data[600+index]
      image = np.clip(image, -1,1)
      image = self.__shift__(image)
      label = 1
      return torch.from_numpy(image).float(), torch.Tensor([label])
    elif self.shift:
      image = self.data[index]
      image = np.clip(image, -1,1)
      if torch.rand(1)<0.51:
        image = self.__shift__(image)
        label = 1
      else:
      #  if torch.rand(1)<0.51:
        image = self.__shift__(image, correct=False)
        label = 0
      #  else:
      #    image = self.__dif_pat__(image, index)
      #  label = 0
      return torch.from_numpy(image).float(), torch.Tensor([label])
    else:
      image = self.data[index]
      image = np.clip(image, -1,1)
      return torch.from_numpy(image).float()

  def __len__(self):
    return self.len
