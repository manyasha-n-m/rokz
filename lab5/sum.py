import numpy as np
import cv2

im = 'data\\image_0{}.png'
mask = 'data\\mask_0{}.png'

images = []
masks = []
for i in range(1,6):
    images.append(cv2.imread(im.format(i)))
masks.append(cv2.imread(mask.format(i), cv2.IMREAD_GRAYSCALE))


class Union:
    def __init__(self, pics, masks, a=1, b=1):
        self.x = np.array(pics)
        self.m = self.x.shape[0]
        self.b = np.array(masks, dtype=bool)
        self.h, self.w = self.b[0].shape
        self.P = np.zeros((self.h, self.w))
        self.q = a*(~self.b)

    def create_g(self):
        norm = {}
        for i in range(self.m-1):
            norm[i, i] = np.zeros((self.h, self.w))
            for j in range(i+1, self.m):
                _norm_kk_ = np.abs(self.x[i, :, :, 0]-self.x[j, :, :, 0]) + \
                            np.abs(self.x[i, :, :, 1]-self.x[j, :, :, 1]) + \
                            np.abs(self.x[i, :, :, 2]-self.x[j, :, :, 2])
                norm[i, j] = norm[j, i] = _norm_kk_
        norm[self.m-1, self.m-1] = np.zeros((self.h, self.w))
        
        for i in range(self.h):
            for j in range(1, self.w):
                pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
