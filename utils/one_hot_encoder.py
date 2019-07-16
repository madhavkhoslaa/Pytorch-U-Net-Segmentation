import skimage
import glob

class HotEncoder():
    def __init__(self, dir, extenstion):
        self.dir= dir
        self.extension= extenstion
        self.color= set()
    def gen_colors(self):
        images= glob.glob(dir+'/*'+ self.extension)
        for img in images:
            image= skimage.io.imread(img)
            h, w= tuple(image.shape())
            for x in range(w):
                for y in range(h):
                    self.color.add(tuple(image[x][y][:]))
        return self.color

    def HotEncode(self):
        pass
