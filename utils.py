import torch
import IPython
import numpy as np
from PIL import Image



def imgpad(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    assert len(img.shape) == 4
    zoers = np.zeros((img.shape[0], img.shape[2], img.shape[3]))
    img = np.concatenate([img, np.expand_dims(zoers, axis=1)], axis=1)
    img = img[:,[0,2,1],:,:]
    mask = np.all(img == 0, axis=1)
    img[:,0,:,:][mask] = 1.0
    img[:,1,:,:][mask] = 1.0
    img[:,2,:,:][mask] = 1.0
    return img


def np2gif(img, gif_path='./event.gif'):
    img = imgpad(img)
    images=[Image.fromarray((img[i]*255).transpose(1, 2, 0).astype(np.uint8)) for i in range(img.shape[0])]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
    IPython.display.display(IPython.display.Image(filename=gif_path))


def load_np2gif(img_path, gif_path='./event.gif'):
    img = np.load(img_path)['arr_0'].astype(np.float32)
    np2gif(img, gif_path=gif_path)