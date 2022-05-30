import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional
class RandomResizedCrop(torch.nn.Module):
    """A RandomResizedCrop reimplementation that does just what we need it to do.
        Crops a section of a PIL image with shape d*img.shape, where
        min_scale/100 <= d <= max_scale/100, at some random coordinate in the image."""

    def __init__(self, min_scale: float = 0.5, max_scale: float = 1) -> None:
        
        super().__init__()
        self.rng = np.random.default_rng() 
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.int_min_scale = int(min_scale * 100)
        self.int_max_scale = int(max_scale * 100)
    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img)
        scale = self.rng.integers(
            low=self.int_min_scale, high=self.int_max_scale) / 100
        (h, w, _) = np_x.shape
        height = int(scale * h)
        width = int(scale * w)
        x_pos = self.rng.random()
        y_pos = self.rng.random()
        y_max = h - height
        x_max = w - width
        left = int(x_pos * x_max)
        top = int(y_pos * y_max)
        img = functional.crop(img, top, left, height, width)
        
        img = functional.resize(img, [h,w]) 
        return img
    def __repr__(self) -> str:
        args = '[{},{}]'.format(self.min_scale, self.max_scale)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'