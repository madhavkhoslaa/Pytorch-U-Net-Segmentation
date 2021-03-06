import os
os.chdir("..")
from utils.one_hot_encoder import HotEncoder
import  numpy as np
import torch
import skimage
from torch.nn.functional import one_hot
img__= skimage.io.imread("/Users/madhav/DataSets/data_semantics/training/semantic_rgb/000034_10.tif")
encoder= HotEncoder(dir= '/Users/madhav/DataSets/data_semantics/training/semantic_rgb', extension="tif", is_binary= False, n_classes= 30)
color_dict= encoder.gen_colors()
onehottt= encoder.PerPixelClassMatrix(img__)
encoded= encoder.HotEncode(onehottt)
ch= encoder.channelSplit(encoded)
for _ in range(30):
    skimage.io.imsave(arr= ch[_].reshape(375,1242), fname= str(_)+".jpeg")
