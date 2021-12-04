import numpy as np
import cv2

im = 'data\\image_0{}.png'
mask = 'data\\mask_0{}.png'

images = []
masks = []
for i in range(1,6):
    images.append(cv2.imread(im.format(i)))
masks.append(cv2.imread(mask.format(i), cv2.IMREAD_GRAYSCALE))


class union:
    def __init__(self, pics, masks, a=1, b=1):
        self.x = np.array(pics)
        self.b = np.array(masks, dtype=bool)
        self.h, self.w = self.b[0].shape
        self.P = np.zeros((self.h, self.w))
        self.q = a*(~self.b)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
