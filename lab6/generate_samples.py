import numpy as np
import cv2
from glob import glob

def e_row(base, idx_row):
    """
    >>> a = np.array([[0,1], [1,2]]); b = np.array([[3,4], [5,6]])
    >>> c = np.array([a, b])
    >>> idx = np.array([0,1,0,0])
    >>> (e_row(c, idx) == np.array([[0,1,3,4,0,1,0,1],[1,2,5,6,1,2,1,2]])).all()
    True
    """
    return np.concatenate(base[idx_row], axis=1)


def main():
    p0 = glob('*e0.png', recursive=True)
    p1 = glob('*e1.png', recursive=True)
    e = np.array([cv2.imread(p0, cv2.IMREAD_GRAYSCALE), cv2.imread(p1, cv2.IMREAD_GRAYSCALE)])
    for i in range(20, 100, 10):
        _im = np.zeros((30, i), dtype=int)
        _base = np.random.randint(0, 2, (3, int(i/10)))
        for j in range(0, 30, 10):
            _im[j:j+10, :] = e_row(e,_base[int(j/10), :])
        cv2.imwrite(f'samples/sample_{i}.png', _im, )

    # real sample
    im = np.zeros((30, 40), dtype=int)
    idx_ = np.array([[1, 0, 1, 1],
                    [1, 0, 0, 1],
                    [0, 1, 0, 0]])
    for j in range(0, 30, 10):
        im[j:j+10, :] = e_row(e, idx_[int(j/10),:])
    cv2.imwrite('samples/sample_0.png', im)


main()

if __name__ == '__main__':
    import doctest
    doctest.testmod()