import skimage
import glob
from tqdm import tqdm
import numpy as np


class HotEncoder():
    def __init__(self,dir, extension, is_binary=False):
        self.dir = dir
        self.extension = extension
        self.is_binary = is_binary
        if is_binary:
            self.color = {(0, 0, 0): 1, (255, 255, 255): 2}
        else:
            self.color = dict()
    
    def gen_colors(self):
        """Iterates through the entire dataset and finds the total colours
            in the images so that they can be used to one hot the image matrix
            for the training data"""
        if self.is_binary:
            return self.color
        else:
            n_color=1
            images = glob.glob(self.dir + '/*.' + self.extension)
            for img in tqdm(images, desc="Generating Color Pallte to Hot Encode"):
                if self.extension=='tif':
                    image= skimage.external.tifffile.imread(img)
                else:
                    image = skimage.io.imread(img)
                shape_ = image.shape
                for x in range(shape_[0]):
                    for y in range(shape_[1]):
                        clr= tuple(image[x][y][:])
                        if clr not in self.color.keys():
                            self.color.update({clr: n_color})
                            n_color+=1
                        else:
                            pass
        return self.color

    def PerPixelClassMatrix(self, Image):
        """Takes an Image and returns a per pixel class
            identification map"""
        shape_= Image.shape
        class_map= np.zeros(shape= (shape_[0], shape_[1]), dtype=float)
        for x in range(shape_[0]):
            for y in range(shape_[1]):
                clr= tuple(Image[x][y][:])[:3]
                class_map[x][y]= self.color[clr]
        return class_map
    def HotEncode(self, class_map):
        """Returns an n channel one hot encoded matrix of the same size as the Image"""
        assert isinstance(class_map, np.ndarray), "Class map has to be an ndarray and not {}".format(type(class_map))
        shape_= class_map.shape
        encoded= np.zeros(shape= (shape_[0], shape_[1], len(self.color)), dtype=float)
        for x in range(shape_[0]):
            for y in range(shape_[1]):
                category= int(class_map[x][y])
                encoded[x][y][category]= 1
        return encoded

    def channelSplit(self,Nch_Hot_encoded):
        """The hot encoded Image is stored as stack of Hot encoded ndarrays
            this function returns a list of cll the channels"""
        return np.dsplit(Nch_Hot_encoded,Nch_Hot_encoded.shape[-1])
    def onehot2classmap(self, one_hot):
        """Method that takes one hot encoded matrix and outputs a classmap generated off the color dictionary"""
        pass
    def classmap2img(self, class_map):
        """Method that takes class_map matrix and gives a matrix like the image"""
        pass