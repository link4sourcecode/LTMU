import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw


def Flip(img, _):  # not from the paper 镜像
    return PIL.ImageOps.mirror(img)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Equalize(img, _): # 均衡图像
    return PIL.ImageOps.equalize(img)


def augment_dict():  # 16 oeprations and their ranges
    l = {
        Equalize: (0, 1),
        Color: (0.1, 1.9),
        Brightness: (0.1, 1.9),
        Sharpness: (0.1, 1.9),
        Flip: (0., 100),
    }

    return l


class SelectAugment:
    def __init__(self, names, m):
        self.names = names
        self.m = m      # [0, 30]
        self.augment_dict = augment_dict()

    def __call__(self, img):
        for name in self.names:
            if name in self.augment_dict:
                if random.random() < 0.5:
                    continue
                op, minval, maxval = name, self.augment_dict[name][0], self.augment_dict[name][1]
                val = (float(self.m) / 30) * float(maxval - minval) + minval
                img = op(img, val)
        return img