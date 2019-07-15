import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss.diceloss import Loss
from torch.utils.data import DataLoader
from hyperparams.hyperparams import hyperparameters
from dataloader.dataloader import ImageLoader, ImageList
from collections import defaultdict
import torch
import torchvision
from torchvision import transforms
import os
from config.config import Config
import scipy.misc


conf= Config('./config.json')
conf= conf.load_conf()
IMAGE_DIR= conf["Train Data"]
ANNOTATIONS_DIR= conf["Annotations Data"]
TEST_DATA= conf["Test Data"]
MODEL_SAVE= conf["Model Save"]
IMAGE_RESOLUTION= tuple(map(int, conf['Resolution'].split(',')))
transforms_compose = transforms.Compose([transforms.Resize(IMAGE_RESOLUTION[:2]),  transforms.ToTensor()])

params = hyperparameters(
    train_percentage=0.6,
    batch_size=1,
    epoch=4,
    n_classes=30)
if torch.cuda.is_available():
    net = UNeT(n_classes= 30, n_channels=3).cuda()
else:
    net = UNeT(n_classes=30, n_channels=3)


Images = ImageList(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=0.1,
    extension="tif")
loss_val = Loss()
Train = ImageLoader(
    data= Images.train_set,
    extension="tif",
    transform=transforms_compose)
Test = ImageLoader(data= Images.test_set, extension="tif", transform=transforms_compose)
TrainLoder = DataLoader(
    Train,
    batch_size=params.hyperparameters["batch_size"],
    shuffle=True)
ValLoader = DataLoader(
    Test,
    batch_size=params.hyperparameters["batch_size"],
    shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(params.hyperparameters["epoch"]):
    metrics = defaultdict()
    running_loss = 0.0
    for i, data in enumerate(TrainLoder, 0):
        if torch.cuda.is_available():
            inputs, labels = data["Image"].cuda(), data["Label"].cuda()
        else:
            inputs, labels = data["Image"], data["Label"]
        if torch.cuda.is_available():
            outputs = net(
                inputs.type(
                    torch.cuda.FloatTensor))
        else:
            outputs = net(
                inputs.type(
                    torch.FloatTensor))
        loss = criterion(input= outputs.view(-1,1,1,1)[:1116000], target=labels.view(-1,1,1,1).type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        print(outputs.size())
        running_loss += loss.item()
        print("Epoch: {} | Loss: {} | Instance: {}".format(int(epoch),loss.item(), i))
        print("Running loss|", running_loss)
torch.save(net.state_dict() , MODEL_SAVE+ "/model.pt")
print("Model saved")
