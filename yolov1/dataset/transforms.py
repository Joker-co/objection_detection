import cv2
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F

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

class Pad(object):
    def __init__(self, pad_value=0, alignment=32):
        self.pad_value = pad_value
        self.alignment = alignment

    def __call__(self, images, image_infos):
        # pad by top-left mode
        max_h, max_w = -1, -1
        for info in image_infos:
            max_h = max(max_h, info[1])
            max_w = max(max_w, info[2])
        target_h = int(np.ceil(max_h / self.alignment) * self.alignment)
        target_w = int(np.ceil(max_w / self.alignment) * self.alignment)
        padded_images = []
        for img in images:
            h, w = img.shape[-2:]
            pad_size = (0, target_w - w, 0, target_h - h)
            padded_images.append(F.pad(img, pad_size, 'constant', self.pad_value).data)
        return torch.stack(padded_images, dim=0)

class ToTensor(object):
    def __call__(self, image, gts):
        image = copy.copy(image)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        gts = torch.from_numpy(gts)
        # transpose image
        image = image.permute(2, 0, 1)
        return image, gts

class Normalize(object):
    def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        self.means = means
        self.stds = stds

    def __call__(self, image):
        image = copy.copy(image)
        image = image / 255
        image = image - torch.tensor(self.means).view(-1, 1, 1)
        image = image / torch.tensor(self.stds).view(-1, 1, 1)
        return image
