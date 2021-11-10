import random

class RandomFlip(object):
    def __call__(self, image, gts):
        if random.random() < 0.5:
            return image, gts

        height, width, _ = image.shape
        flip_x1 = width - gts[:, 2]
        gts[:, 2] = width - gts[:, 0]
        gts[:, 0] = flip_x1
        image = image[:, ::-1, :]
        return image, gts