import torch
import torch.nn as nn
import math
import numpy as np
import tqdm
dtype = torch.float

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, running on CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available, running on MPS device")
else:
    device = torch.device("cpu")
    print("MPS is not available, running on CPU")

from model import network
