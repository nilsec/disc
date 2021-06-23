import numpy as np
import os
from PIL import Image

def flatten_image(pil_image):
    """
    pil_image: image as returned from PIL Image
    """
    return np.array(pil_image[:,:,0], dtype=np.float32)

def normalize_image(image):
    """
    image: 2D input image
    """
    return (image.astype(np.float32)/255. - 0.5)/0.5

def open_image(image_path, flatten=True, normalize=True):
    im = np.asarray(Image.open(image_path))
    if flatten:
        im = flatten_image(im)
    if normalize:
        im = normalize_image(im)
    return im

def save_image(array, image_path, norm=False):
    if norm:
        array = array.astype(np.float)
        array/=np.max(np.abs(array))
        array *= 255
        array = array.astype(np.uint8)

    im = Image.fromarray(array)
    im = im.convert('RGB')
    im.save(image_path)

