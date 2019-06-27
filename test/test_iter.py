import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss import diceloss
from torch.utils.data import DataLoader
from hyperparams.hyperparams import hyperparameters
from dataloader.dataloader import ImageLoader, TrainSet, TestSet
params = hyperparameters(train_percentage=0.6, batch_size=4)
net = UNeT(n_class=1)
IMAGE_DIR = "/Users/madhav/DataSets/AerialImageDataset/train/images/*.tif"
ANNOTATIONS_DIR = "/Users/madhav/DataSets/AerialImageDataset/train/gt/*.tif"
Images = ImageLoader(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=params.hyperparameters['train_percentage'])
Train = TrainSet(Images.train_set, extension="tif", transform=None)
Test = TestSet(Images.test_set, extension="tif", transform=None)
print(Images.test_set)
for i in range(len(Train)):
    print(Train[i])