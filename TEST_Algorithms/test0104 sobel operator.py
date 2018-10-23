import numpy as np
from skimage import data, filters
import matplotlib.pyplot as plt
from scipy.misc import *

def rgb2grey(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

testmMat = imread('image/test.png')
test = rgb2grey(testmMat)

my_new = filters.sobel(test)

plt.imshow(my_new)
plt.show()

