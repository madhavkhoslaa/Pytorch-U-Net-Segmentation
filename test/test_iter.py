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
TrainLoder = DataLoader(Train, batch_size=4)
for i, data in enumerate(TrainLoder, 0):
    inputs, labels = data["Image"], data["Label"]
    print(inputs.size())
