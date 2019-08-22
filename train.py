import torch.nn as nn
import torch.optim as optim
from model.Unet import UNeT
from loss.diceloss import Loss
from torch.utils.data import DataLoader
from hyperparams.hyperparams import hyperparameters
from dataloader.dataloader import ImageLoader, ImageList
from collections import defaultdict
import torch
from torchvision import transforms
from config.config import Config
from tqdm import tqdm
from utils.one_hot_encoder import HotEncoder


conf = Config('config.json')
conf = conf.load_conf()
IMAGE_DIR = conf["Train Data"]
ANNOTATIONS_DIR = conf["Annotations Data"]
TEST_DATA = conf["Test Data"]
MODEL_SAVE = conf["Model Save"]
IMAGE_RESOLUTION = tuple(map(int, conf['Resolution'].split(',')))
transforms_compose = transforms.Compose(
    [transforms.ToTensor()])

params = hyperparameters(
    train_percentage=0.6,
    batch_size=1,
    epoch=4,
    n_classes=1)
if torch.cuda.is_available():
    net = UNeT(n_classes=29, n_channels=3).cuda()
else:
    net = UNeT(n_classes=29, n_channels=3)
encoder= HotEncoder(is_binary= False, dir= ANNOTATIONS_DIR, extension="png")
color_dict= encoder.gen_colors()
Images = ImageList(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=0.9,
    extension="png")
loss_val = Loss()
Train = ImageLoader(
    encoder_obj= encoder,
    data=Images.train_set,
    extension="png",
    transform=transforms_compose)
Test = ImageLoader(
    encoder_obj= encoder,
    data=Images.test_set,
    extension="tifngng",
    transform=transforms_compose)
TrainLoader = DataLoader(
    Train,
    batch_size=params.hyperparameters["batch_size"],
    shuffle=True)
ValLoader = DataLoader(
    Test,
    batch_size=params.hyperparameters["batch_size"],
    shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for i, data in enumerate(TrainLoader, 0):
    inputs, labels= data["Image"].permute(0, 3, 1, 2), data["Label"].permute(0, 3, 1, 2)
    print(inputs, labels)

for epoch in tqdm(
        range(params.hyperparameters["epoch"]), desc="Training Loop"):
    metrics = defaultdict()
    running_loss = 0.0
    for i, data in enumerate(TrainLoader, 0):
        if torch.cuda.is_available():
            inputs, labels = data["Image"].cuda(), data["Label"].cuda()
        else:
            inputs, labels = data["Image"], data["Label"]
        outputs = net(inputs)
        loss = criterion(input=outputs,target=labels)
        #There was a dimension strip above here check if you need it again, lad.
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(
            "Epoch: {} | Loss: {} | Instance: {}".format(
                int(epoch),
                loss.item(),
                i))
        print("Running loss|", running_loss)
torch.save(net.state_dict(), MODEL_SAVE + "/model.pt")
print("Model saved")