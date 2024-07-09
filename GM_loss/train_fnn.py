import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
random.seed(577)

import numpy as np
np.random.seed(577)
import pandas as pd

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.optim as optim

from Module.gm_loss import GM_Loss
from Models.cnn import FNN


from torch.utils.data import DataLoader, random_split


import torch.nn as nn

import copy

import warnings  
warnings.filterwarnings("ignore")



