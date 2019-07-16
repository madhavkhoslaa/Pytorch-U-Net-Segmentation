import skimage
import glob
from tqdm import tqdm


class HotEncoder():
    def __init__(self, dir, extension, is_binary= True):
        self.dir= dir
        self.extension= extension
        self.is_binary= is_binary
        if is_binary:
            self.color= set(((255, 255, 255), (0, 0, 0)))
        else:
            self.color= set()

    def gen_colors(self):
        """Iterates through the entire dataset and finds the total colours
            in the images so that they can be used to one hot the image matrix 
            for the training data"""
        if self.is_binary:
            return self.color
        images= glob.glob(self.dir+'/*.'+ self.extension)
        for img in tqdm(images, desc="Generating Color Pallte to Hot Encode"):
            image= skimage.io.imread(img)
            shape_= image.shape
            for x in range(shape_[0]):
                for y in range(shape_[1]):
                    self.color.add(tuple(image[x][y][:]))
        return self.color

    def HotEncode(self):
        pass
