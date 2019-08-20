
## This repository is an implementation on the [U-Net Paper](https://arxiv.org/abs/1505.04597).  
 ![](https://i.imgur.com/EHDpics.png)

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/madhavkhoslaa/Pytorch-U-Net-Segmentation" data-icon="octicon-star" data-size="large" aria-label="Star madhavkhoslaa/Pytorch-U-Net-Segmentation on GitHub">Star</a>

* The script/getdataset.sh downloads the [The Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/).This is not the best data set if you want to do aerial image segmentation on your own computer, because of computational limits because of 5k Images.
* This Repo contains a hot encoder which encodes the dataset of n classes to a one hot encoded matrix of n* height* width for every class. The encoder can be seen at utils/ .
* To use train it, put the Images and the gt file locations in the config.json folder and execute train.py, a .pth file will be saved for the following model in the working directory.
>Ch/ foler includes the example for creating a N class hot encoded image
 
> If you feel any need to contribute don’t feel shy to do a [pull request](https://github.com/madhavkhoslaa/U-Net-Segmentation/pulls) or [contact me](mailto:madhavkhosla@cock.li):D

> [To Understand Repo structure](https://veniversum.me/git-visualizer/?owner=madhavkhoslaa&repo=Pytorch-U-Net-Segmentation) 

>Credits: [milesial's UNET Repository](https://github.com/milesial/Pytorch-UNet)
