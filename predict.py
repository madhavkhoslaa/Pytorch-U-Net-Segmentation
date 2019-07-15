import torch
import torchvision
from model.Unet import UNeT
import sys
import torchvision

to_pred= sys.argv[1:]
model= UNeT(n_channels= 3, n_classes=30)
model.load_state_dict('./model.pt')
model.eval()
for _ in to_pred:
    output= model(_)
    torchvision.utils.save_image(output, _+'prediction.jpeg')
    print("Output saved at {}+prediction.jpeg".format(_))