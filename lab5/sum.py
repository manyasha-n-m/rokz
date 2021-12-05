import numpy as np
import cv2

im_path = 'data\\image_0{}.png'
mask_path = 'data\\mask_0{}.png'
if not __name__ == "__main__":
    im_path = 'lab5/data/image_0{}.png'
    mask_path = 'lab5/data/mask_0{}.png'


def read(im, mask):
    images = []
    masks = []
    for i in range(1, 6):
        images.append(cv2.imread(im.format(i)))
        masks.append(cv2.imread(mask.format(i), cv2.IMREAD_GRAYSCALE))
    return images, masks


class Union:
    """
    >>> images, masks = read(im_path, mask_path)
    >>> u = Union(images, masks)
    >>> q = u.q[:,:,-1]; q.shape
    (5, 1000)
    >>> g = u.g[:,:,:,-1]; g.shape
    (5, 5, 1000)
    >>> f0 = np.min(q+g, axis=1); f0.shape
    (5, 1000)
    """
    def __init__(self, pics, masks, alpha=1, beta=1):
        self.x = np.array(pics)
        self.m = self.x.shape[0]
        self.b = np.array(masks, dtype=bool)
        self.h, self.w = self.b[0].shape
        self.P = np.zeros((self.h, self.w))
        self.alpha, self.beta = alpha, beta
        self.q = alpha*(~self.b)
        self.g = self.create_g()

    def create_g(self):
        g = np.zeros((self.m, self.m, self.h, self.w-1))
        for i in range(self.m-1):
            for j in range(i+1, self.m):
                _norm_kk_ = np.abs(self.x[i, :, :, 0]-self.x[j, :, :, 0]) + \
                            np.abs(self.x[i, :, :, 1]-self.x[j, :, :, 1]) + \
                            np.abs(self.x[i, :, :, 2]-self.x[j, :, :, 2])
                g[i, j, :, :] = g[j, i, :, :] = _norm_kk_[:, :-1] + _norm_kk_[:, 1:]
        return g

    def calc_f(self):
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
