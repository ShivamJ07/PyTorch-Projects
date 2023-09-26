import os
import torch
from torch.utils.data import dataset
from skimage import io

!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

!tar -xzf food-101.tar.gz

target_path = os.path.join(os.getcwd(), 'images')
foods = sorted(os.listdir(target_path))
foods_to_int = {}
for i, food in enumerate(foods):
  foods_to_int[food] = i