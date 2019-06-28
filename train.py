import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss.diceloss import Loss
from torch.utils.data import DataLoader
from hyperparams.hyperparams import hyperparameters
from dataloader.dataloader import ImageLoader, TrainSet, TestSet
from collections import defaultdict
import torch
params = hyperparameters(train_percentage=0.6, batch_size=10, epoch=4)
if torch.cuda.is_available():
    net= UNeT(n_class=1).cuda()
else:
    net= net= UNeT(n_class=1)

IMAGE_DIR = "/Users/madhav/DataSets/AerialImageDataset/train/images/*.tif"
ANNOTATIONS_DIR = "/Users/madhav/DataSets/AerialImageDataset/train/gt/*.tif"
Images = ImageLoader(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=0.7)
loss_val = Loss()
Train = TrainSet(Images.train_set, extension="tif", transform=None)
Test = TestSet(Images.test_set, extension="tif", transform=None)
TrainLoder = DataLoader(
    Train,
    batch_size=params.hyperparameters["batch_size"],
    shuffle=True)
ValLoader = DataLoader(
    Test,
    batch_size=params.hyperparameters["batch_size"],
    shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(params.hyperparameters["epoch"]):
    metrics = defaultdict()
    running_loss = 0.0
    for i, data in enumerate(TrainLoder, 0):
        inputs, labels= data["Image"], data["Label"]
        inputs, labels= inputs.transpose(0, 3).transpose(1, 2), labels.transpose(0, 3).transpose(1, 2)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_val.calc_loss(outputs, labels, metrics, bce_weight=0.5)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
        (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.

print('Finished Training')
