import skimage
import glob
from tqdm import tqdm
import numpy as np


class HotEncoder():
    def __init__(self, dir, extension, is_binary= False):
        self.dir = dir
        self.extension = extension
        self.is_binary= is_binary
        if self.is_binary:
            self.color = {(0, 0, 0): 0, (255, 255, 255): 1}
        else:
            self.color = dict()

    def gen_colors(self):
        """Iterates through the entire dataset and finds the total colours
            in the images so that they can be used to one hot the image matrix
            for the training data"""
        if self.is_binary:
            return self.color
        else:
            n_color = 1
            images = glob.glob(self.dir + '/*.' + self.extension)
            for img in tqdm(
                    images, desc="Generating Color Pallte to Hot Encode"):
                if self.extension == 'tif':
                    image = skimage.external.tifffile.imread(img)
                else:
                    image = skimage.io.imread(img)
                shape_ = image.shape
                for x in range(shape_[0]):
                    for y in range(shape_[1]):
                        clr = tuple(image[x][y][:])
                        if clr not in self.color.keys():
                            self.color.update({clr: n_color})
                            n_color += 1
                        else:
                            pass
        return self.color

    def PerPixelClassMatrix(self, Image, color_dict=None):
        """Takes an Image and returns a per pixel class
            identification map"""
        if color_dict is None:
            color_dict = self.color
        else:
            assert isinstance(color_dict, dict)
        shape_ = np.array(Image).shape
        class_map = np.zeros(shape=(shape_[0], shape_[1]), dtype=float)
        for x in range(shape_[0]):
            for y in range(shape_[1]):
                clr = tuple(Image[x][y][:])[:3]
                class_map[x][y] = color_dict[clr]
        return np.array(class_map)

    def hotEncode(self, class_map):
        """Inputs the class map of the image and returns the hot encoded
        N channel output. Each channel represents the presence of a class per pixel"""
        assert isinstance(
            class_map, np.ndarray), "Class map has to be an ndarray and not {}".format(
            type(class_map))
        shape_ = class_map.shape
        encoded = np.zeros(
            shape=(
                shape_[0],
                shape_[1],
                len(self.color)),
            dtype=float)
        for x in range(shape_[0]):
            for y in range(shape_[1]):
                category = int(class_map[x][y])
                try:
                    encoded[x][y][ca] = 255
                except:
                    encoded[x][category] = 255
        return np.array(encoded)

    def channelSplit(self, Nch_Hot_encoded):
        """The hot encoded Image is stored as stack of Hot encoded ndarrays
            this function returns a list of cll the channels"""
        return np.dsplit(Nch_Hot_encoded, Nch_Hot_encoded.shape[-1])
    def OneHot2ClassMap(self, one_hot_encoded):
        pass

    def ClassMap2RGB(self, class_map):
        pass





