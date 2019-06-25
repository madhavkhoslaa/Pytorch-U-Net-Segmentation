import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss import diceloss
from torch.utils.data import DataLoader
from dataloader.dataloader import ImageLoader, TrainSet, TestSet
net= UNeT(n_class=1)
DATASET="/path/to/dataset"

Images= ImageLoader(Images="/Path/to/images", Annotations="/Path/to/Annotations", train_percentage= 0.6)
Train= TrainSet(Images.train_set, extension="JPEG", transform=None)
Test= TestSet(Images.test_set, extension="JPEG", transform=None)

Train_Loader= DataLoader(Train, batch_size=4, shuffle=True)
Test_Loadrt= DataLoader(Test, batch_size=4, shuffle=True)

#TODO: Train Loop