import time
st = time.time()
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
    >>> flast = np.min(q+g, axis=1); flast.shape
    (5, 1000)
    >>> f = u.calc_f()
    >>> a = np.array([[1,1], [2,3], [4,2]]); np.argmin(a, axis=1).astype(int)
    array([0, 0, 1])
    """
    def __init__(self, pics, masks, alpha=1., beta=1.):
        self.x = np.array(pics)
        self.m = self.x.shape[0]
        self.b = np.array(masks, dtype=bool)
        self.h, self.w = self.b[0].shape
        self.alpha, self.beta = alpha, beta
        self.q = self.alpha*(~self.b)
        self.g = self.create_g()

    def create_g(self):
        g = np.zeros((self.m, self.m, self.h, self.w))
        for i in range(self.m-1):
            for j in range(i+1, self.m):
                _norm_kk_ = np.abs(self.x[i, :, :, 0]-self.x[j, :, :, 0]) + \
                            np.abs(self.x[i, :, :, 1]-self.x[j, :, :, 1]) + \
                            np.abs(self.x[i, :, :, 2]-self.x[j, :, :, 2])
                g[i, j, :, 1:] = g[j, i, :, 1:] = self.beta*(_norm_kk_[:, :-1] + _norm_kk_[:, 1:])
        return g

    def calc_f(self):
        w = self.w
        f = np.zeros((w, self.m, self.h))
        for j in range(w-1, 0, -1):
            f[j-1, :, :] = np.min(self.q[:, :, j] + f[j, :, :] + self.g[:, :, :, j], axis=1)
        return f

    def optimal(self):
        f = self.calc_f()
        K = np.zeros((self.w, self.h), dtype=int)
        K[0, :] = np.argmin(self.q[:, :, 0] + f[0, :, :], axis=0)
        for j in range(1, self.w):
            _table = self.q[:, :, j] + self.g_optimal(K[j-1, :], j) + f[j, :, :]
            K[j, :] = np.argmin(_table, axis=0)
        return K

    def merge(self):
        K = self.optimal()
        return np.take_along_axis(self.x, K.T[None, :, :, None], axis=0)[0]

    def g_optimal(self, K, j):
        return np.take_along_axis(self.g[:, :, :,j], K[None, None, :], axis=0)[0]


im, m = read(im_path, mask_path)
u = Union(im, m, 255*8)
res = u.merge()
cv2.imwrite('18.png', res)
print(time.time()-st)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
