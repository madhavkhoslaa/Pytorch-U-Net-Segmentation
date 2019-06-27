from torch.utils.data import Dataset
import glob
import skimage
import torch

class ImageLoader():
    def __init__(self, Images, Annotations,
                 train_percentage):
        self.Image = glob.glob(Images)
        self.Annotations = glob.glob(Annotations)
        self.train_percentage = train_percentage
        train_len = int(train_percentage * len(Images))
        self.train_set = {"Images": self.Image[:train_len],
                          "Annotations": self.Annotations[:train_len]}
        self.test_set = {"Images": self.Image[train_len:],
                         "Annotations": self.Annotations[train_len:]}


class TrainSet(Dataset):
    def __init__(self, train_data, extension="jpeg", transform=None):
        self.extension = extension.lower()
        self.transform = transform
        self.images = train_data["Images"]
        self.target_images = train_data["Annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.extension == "png":
            image = skimage.io.imread(self.images[index])[:3]
            label = skimage.io.imread(self.target_images)[:3]
        if self.extension == "tif":
            image = skimage.external.tifffile.imread(self.images[index])
            label = skimage.external.tifffile.imread(self.target_images[index])
        else:
            image = skimage.io.imread(self.images[index])
            label = skimage.io.imread(self.target_images[index])
        if self.transform:
            image = self.transform(image)
        if torch.cuda.is_available():
            return {"Image": torch.from_numpy(image).permute(0, 3, 1, 2).cuda(), "Label": torch.from_numpy(label).permute(0, 3, 1, 2).cuda()}
        else:
            return {"Image": torch.from_numpy(image).permute(0, 3, 1, 2), "Label": torch.from_numpy(label).permute(0, 3, 1, 2)}


class TestSet(Dataset):
    def __init__(self, train_data, extension="jpeg", transform=None):
        self.extension = extension.lower()
        self.transform = transform
        self.images = train_data["Images"]
        self.target_images = train_data["Annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.extension == "png":
            image = skimage.io.imread(self.images[index])[:3]
            label = skimage.io.imread(self.target_images)[:3]
        if self.extension == "tif":
            image = skimage.external.tifffile.imread(self.images[index])
            label = skimage.external.tifffile.imread(self.target_images[index])
        else:
            image = skimage.io.imread(self.images[index])
            label = skimage.io.imread(self.target_images[index])
        if self.transform:
            image = self.transform(image)
        if torch.cuda.is_available():
            return {"Image": torch.from_numpy(image).permute(0, 3, 1, 2).cuda(), "Label": torch.from_numpy(label).permute(0, 3, 1, 2).cuda()}
        else:
            return {"Image": torch.from_numpy(image).permute(0, 3, 1, 2), "Label": torch.from_numpy(label).permute(0, 3, 1, 2)}
