
## This repository is an implementation on the [U-Net Paper](https://arxiv.org/abs/1505.04597).  
 ![](https://i.imgur.com/EHDpics.png)

* The script/getdataset.sh downloads the [The Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/).This is not the best data set if you want to do aerial image segmentation on your own computer, because of computational limits because of 5k Images.
* This Repo contains a hot encoder which encodes the dataset of n classes to a one hot encoded matrix of n* height* width for every class. The encoder can be seen at utils/ .
* To use train it, put the Images and the gt file locations in the config.json folder and execute train.py, a .pth file will be saved for the following model in the working directory.
* Ch/ foler includes the example for creating a N class hot encoded image

# Working
* git clone https://github.com/madhavkhoslaa/Pytorch-U-Net-Segmentation
* Open the config.json file and put the location of your dataset in the json files and put the resolution of the Images(you cannot train model with different sized images and a single batch size because of PyTorch collate)
* Then edit the train.py file and esit the file extensions.
* Train the model, by running train.py
### Using just the hot encoder
* img= skimage.io.imread("someimage.jpeg")
> Load your Image
* encoder= HotEncoder(dir= '', extension="tif", is_binary= False)
> Make an encoder object for you dataset.
* classes= encoder.gen_colors()
> This generates the number of classes from the number of colors in your annotated data.
* ClassMatrix= encoder.PerPixelClassMatrix(image)
> This method returns a martrix of the same size height and width but single channel, depecting class of each pixel in the Image.
* encoded= encoder.HotEncode(ClassMatrix)
> This encodes your Image to n channel matrix of same height and width
>You also might want to view each channel depecting every class for an Image, to split the n channel matrix to n single channel matrices.
* ch= encoder.channelSplit(encoded)
> This returns n number of matrices depecting each class in an Image.
This method is also used to encode the Image in the dataloader.


# Known Issues
* Single Channel tif files cannot be hot encoded right now. If you have a single channel tif file for the annotation, convert it into a png or a 3 channel tif file.
* Predict.py is not tested.

# ___ 
> If you feel any need to contribute don’t feel shy to do a [pull request](https://github.com/madhavkhoslaa/U-Net-Segmentation/pulls) or [contact me](mailto:madhavkhosla@cock.li) :D

> [To Understand Repo structure](https://veniversum.me/git-visualizer/?owner=madhavkhoslaa&repo=Pytorch-U-Net-Segmentation) 

# Credits: 
[milesial's UNET Repository for the U-Net Model](https://github.com/milesial/Pytorch-UNet)
