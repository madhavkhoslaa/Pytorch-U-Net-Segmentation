import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss import diceloss
from torch.utils.data import DataLoader
from hyperparams.hyperparams import hyperparameters
from dataloader.dataloader import ImageLoader, TrainSet, TestSet
from loss.diceloss import dice_loss, calc_loss
from collections import defaultdict


params = hyperparameters(train_percentage=0.6, batch_size=4)
net = UNeT(n_class=1)
IMAGE_DIR = "~/DataSets/AerialImageDataset/train/images/"
ANNOTATIONS_DIR = "~/DataSets/AerialImageDataset/train/gt/"
Images = ImageLoader(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=params.hyperparameters['train_percentage'],
    extension="tif")
Train = TrainSet(Images.train_set, extension="tif", transform=None)
Test = TestSet(Images.test_set, extension="tif", transform=None)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(4):
    metrics= defaultdict()
    running_loss = 0.0
    for i, data in enumerate(Train, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = calc_loss(outputs, labels, metrics,bce_weight=0.5)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')