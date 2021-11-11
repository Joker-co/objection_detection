import cv2
import copy
import numpy as np
import random

class RandomFlip(object):
    def __call__(self, image, gts):
        image, gts = copy.copy(image), copy.copy(gts)
        if random.random() < 0.5:
            return image, gts

        height, width, _ = image.shape
        flip_x1 = width - gts[:, 2]
        gts[:, 2] = width - gts[:, 0]
        gts[:, 0] = flip_x1
        image = image[:, ::-1, :]
        return image, gts

class RandomResize(object):
    def __init__(self, scales=[416], max_scale=416):
        self.scales = scales
        self.max_scale = max_scale

    def get_scale_factor(self, height, width):
        # random select scale
        scale = np.random.choice(self.scales)
        long_side, short_side = max(height, width), min(height, width)
        return min(self.max_scale / long_side, scale / short_side)

    def __call__(self, image, gts):
        image, gts = copy.copy(image), copy.copy(gts)
        h, w = image.shape[:2]
        scale_factor = self.get_scale_factor(h, w)

        # resize_img
        resize_h, resize_w = int(h * scale_factor), int(w * scale_factor)
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        # resize gts
        gts[:, :4] = gts[:, :4] * scale_factor
        return image, gts, scale_factor