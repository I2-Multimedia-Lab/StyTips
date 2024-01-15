import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
import torchvision.transforms as T


########################################## Dataset ##########################################

class SimpleDataset(Dataset):
  def __init__(self, dir_path, transforms=None):
    super(SimpleDataset, self).__init__()
    assert os.path.isdir(dir_path), """'dir_path' needs to be a directory path."""
    self.dir_path = dir_path
    self.img_paths = os.listdir(self.dir_path)
    
    if not transforms is None:
      self.transforms = transforms
    else:
      self.transforms = T.ToTensor()

  def __getitem__(self, index):
    file_name = self.img_paths[index]
    img = Image.open(os.path.join(self.dir_path, file_name)).convert('RGB')
    img = self.transforms(img)
    return img

  def __len__(self):
    return len(self.img_paths)




########################################## Sampler ##########################################

def InfiniteSampler(n):
  """ Generator returning the random number between 0 to n-1
  """
  i = n - 1
  order = np.random.permutation(n)
  while True:
    yield order[i]
    i += 1
    if i >= n:
      np.random.seed()
      order = np.random.permutation(n)
      i = 0

class InfiniteSamplerWrapper(Sampler):
  def __init__(self, data_source):
    self.num_samples = len(data_source)

  def __iter__(self):
    return iter(InfiniteSampler(self.num_samples))

  def __len__(self):
    return 2 ** 31
