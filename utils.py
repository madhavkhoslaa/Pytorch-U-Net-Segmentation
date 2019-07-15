import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.dataloader import ImageLoader, ImageList
import torch
from torchvision import transforms
from config.config import Config


conf= Config('./config.json')
conf= conf.load_conf()
IMAGE_DIR= conf["Train Data"]
ANNOTATIONS_DIR= conf["Annotations Data"]
TEST_DATA= conf["Test Data"]
MODEL_SAVE= conf["Model Save"]
IMAGE_RESOLUTION= tuple(map(int, conf['Resolution'].split(',')))
transforms_compose = transforms.Compose([transforms.Resize(IMAGE_RESOLUTION[:2]),  transforms.ToTensor()])

Images = ImageList(
    Images=IMAGE_DIR,
    Annotations=ANNOTATIONS_DIR,
    train_percentage=0.1,
    extension="tif")
Train = ImageLoader(
    data= Images.train_set,
    extension="tif",
    transform=transforms_compose)
TrainLoder = DataLoader(
    Train,
    batch_size=1,
    shuffle=True)
def get_out_size(model):
    """Iterates through the dataset like in training but does not calculate loss
        does not backprop just calculates the output size."""
    for epoch in range(1):
        for i, data in enumerate(TrainLoder, 0):
            if torch.cuda.is_available():
                inputs, labels = data["Image"].cuda(), data["Label"].cuda()
            else:
                inputs, labels = data["Image"], data["Label"]
            del labels
            if torch.cuda.is_available():
                outputs = model(
                    inputs.type(
                        torch.cuda.FloatTensor))
            else:
                outputs = model(
                    inputs.type(
                        torch.FloatTensor))
            return outputs.size()

