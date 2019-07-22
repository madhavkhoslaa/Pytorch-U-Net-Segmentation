from utils.one_hot_encoder import HotEncoder
import  numpy as np
import torch
import skimage
img__= skimage.io.imread("/Users/madhav/Desktop/Screenshot 2019-07-22 at 12.44.30 PM.png")
encoder= HotEncoder(dir= "/Users/madhav/Desktop/", extension="png", is_binary= False)
color_dict= encoder.gen_colors()
#onehottt= encoder.PerPixelClassMatrix(img__)










def masks_to_colorimg(masks, colors):
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape
    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]
            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)
def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

hot1= rgb_to_onehot(img__[:3], color_dict)
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

