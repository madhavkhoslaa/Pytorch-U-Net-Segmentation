import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss import diceloss
from torch.utils.data import DataLoader
from dataloader.dataloader import ImageLoader, TrainSet, TestSet
net= UNeT(n_class=1)
DATASET="/path/to/dataset"

Images= ImageLoader(Images="/Path/to/images", Annotations="/Path/to/Annotations", train_percentage= 0.6)
trainloader= TrainSet


