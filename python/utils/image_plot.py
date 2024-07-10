import skimage.exposure as exposure
import numpy as np
from matplotlib.colors import ListedColormap

def render_color(img, vmax=None):
    # norlaization
    if vmax == None:
        vmax = np.array([np.percentile(img[..., 0], 99),\
                         np.percentile(img[..., 1], 99)])
        # vmax = np.percentile(img, 99.99)
        # vmax = np.percentile(img, 99)
        # vmax = np.max(img)
    img = img / vmax
    img = np.clip(img, 0.0, 1.0)

    # gamma correction
    # img = exposure.adjust_gamma(img, gamma=0.5)

    # set colors
    color_magenta = np.asarray([255, 0, 255]).reshape((1, 1, -1)) # 642
    color_green   = np.asarray([0, 255, 0]).reshape((1, 1, -1))   # 560

    img_color = (img[..., 0][..., None] * color_magenta\
               + img[..., 1][..., None] * color_green)
    img_color = np.clip(img_color, 0, 255)

    return img_color.astype(np.uint8)

def normalization(img_gray, vmin, vmax):
    """Normalize image to 0-1.
    """
    vmin, vmax = np.array(vmin), np.array(vmax)
    img_norm = (img_gray - vmin)/(vmax - vmin)
    img_norm = np.clip(img_norm, 0, 1)
    return img_norm

def look_up(img_gray, lut, rgb_type='rgb'):
    """Apply LUT to numpy array.""" 
    img_gray_flat = img_gray.reshape(-1)
    rgb = lut(img_gray_flat)  
    rgb = rgb.reshape(img_gray.shape+(4,))
    rgb = rgb[...,:-1]
    if rgb_type == 'bgr':
        rgb = np.flip(rgb, axis=-1)
    return rgb

def merge(imgs):
    assert imgs.__len__() >1, "imgs shoud be an array with at least two images"
    merged = 0
    for img in imgs:
        merged += img.astype(np.float64)
    merged = merged*255.
    merged[merged>255] = 255
    return merged.astype(np.uint8)

def render(img, cmaps=['gray'], vmin=None, vmax=None, rgb_type='rgb'):
    """The last dimention should be the channel.
    """
    num_channel = img.shape[-1]
    assert num_channel == len(cmaps), 'One cmap for each channel'

    if vmax == None:
        vmax = []
        for i in range(num_channel):
            vmax.append(np.percentile(img[..., i], 100))
        vmax = np.array(vmax)

    if vmin == None:
        vmin = []
        for i in range(num_channel):
            vmin.append(np.percentile(img[..., i], 0))
        vmin = np.array(vmin)

    #Normalise images to defined value (0-255).
    img_norm = normalization(img, vmin, vmax)

    #Grayscale images converted to rgb with a LUT
    imgs = []
    for i in range(num_channel):
        # img_sc = exposure.adjust_gamma(img_norm[...,i], gamma=2.0)
        img_sc = img_norm[...,i]
        imgs.append(look_up(img_sc, cmaps[i], rgb_type=rgb_type))

    img_merged = merge(imgs)
    return img_merged