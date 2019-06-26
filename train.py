import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss import diceloss
from torch.utils.data import DataLoader
from hyperparams.hyperparams import hyperparameters
from dataloader.dataloader import ImageLoader, TrainSet, TestSet
params = hyperparameters(train_percentage=0.6, batch_size=4)
net = UNeT(n_class=1)
IMAGE_DIR = "/home/fatjuicyboi/DataSets/AerialImageDataset/train/images/"
ANNOTATIONS_DIR = "/home/fatjuicyboi/DataSets/AerialImageDataset/train/gt/"
Images = ImageLoader(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=params.hyperparameters['train_percentage'],
    extension="tif")
Train = TrainSet(Images.train_set, extension="tif", transform=None)
Test = TestSet(Images.test_set, extension="tif", transform=None)

for i in range(len(Train)):
    print(Train[i])