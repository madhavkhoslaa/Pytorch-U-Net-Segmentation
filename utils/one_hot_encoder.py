import skimage
import glob
from tqdm import tqdm


class HotEncoder():
    def __init__(self, dir, extension, is_binary=True):
        self.dir = dir
        self.extension = extension
        self.is_binary = is_binary
        if is_binary:
            self.color = set(((255, 255, 255), (0, 0, 0)))
        else:
            self.color = dict()

    def gen_colors(self):
        """Iterates through the entire dataset and finds the total colours
            in the images so that they can be used to one hot the image matrix
            for the training data"""
        if self.is_binary:
            return self.color
        images = glob.glob(self.dir + '/*.' + self.extension)
        for img in tqdm(images, desc="Generating Color Pallte to Hot Encode"):
            image = skimage.io.imread(img)
            shape_ = image.shape
            n_color=1
            for x in range(shape_[0]):
                for y in range(shape_[1]):
                    clr= tuple(image[x][y][:])
                    if clr not in self.color.keys():
                        self.color.update({clr: n_color})
                        n_color+=1
                    else:
                        pass
        return self.color

    def gen_class_image(self):
        pass
