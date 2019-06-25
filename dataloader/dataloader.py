from torch.utils.data import Dataset
import glob
import skimage


class ImageLoader(self):
    def __init__(self, Images, Annotations, train_percentage):
        self.Image = glob.glob(Images)
        self.Annotations = glob.glob(Images)
        self.train_percentage = train_percentage
        train_len = int(train_percentage * len(Images))
        self.train_set = {"Images": self.Image[:train_len],
                        "Annotations": self.Annotations[:train_len]}
        self.test_set = {"Images": self.Image[train_len:], 
                        self.Annotations[train_len:]}


class TrainSet(Dataset):
    def __init__(self, train_data, extension="jpeg", transform):
        self.extension = extension
        self.transform = transform
        self.images = train_data["Images"]
        self.target_images = train_data["Annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.extension == "png":
            image = skimage.io.imread(self.images[index])[:3]
            label = skimage.io.imread(self.target_images)[:3]
        else:
            image = skimage.io.imread(self.images[index])
            label = skimage.io.imread(self.target_images)
        if self.transform:
            image = self.transform(image)
        return {"Image": image, "Label":  label}


class TestSet(Dataset):
    def __init__(self, test_data, extension="jpeg", transform):
        self.extension = extension
        self.transform = transform
        self.images = test_data["Images"]
        self.target_images = test_data["Annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.extension == "png":
            image = skimage.io.imread(self.images[index])[:3]
            label = skimage.io.imread(self.target_images)[:3]
        else:
            image = skimage.io.imread(self.images[index])
            label = skimage.io.imread(self.target_images)
        if self.transform:
            image = self.transform(image)
        return {"Image": image, "Label":  label}
