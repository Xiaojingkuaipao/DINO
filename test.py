# %%
import torch
import torchvision.transforms as T
from torch import nn
from util.misc import NestedTensor
from PIL import Image

img = Image.open('./figs/idea.jpg')
transform = T.Compose([
    T.ToTensor()
])

img_tensor = transform(img)
print(img_tensor.shape)

nested_tensor = NestedTensor(img_tensor.unsqueeze(0), 'auto')

print(nested_tensor.shape)

image_size = nested_tensor.imgsize()
print(image_size)

img_list = nested_tensor.to_img_list()

img_tensor_no_padding = img_list[0]

img_no_padding = T.ToPILImage()(img_tensor_no_padding)

img_no_padding.show()

print(nested_tensor.mask[0])


