import numpy as np
from scipy.misc import *

testmMat = imread('image/Lion.jpg')
#testmMat -= np.min(testmMat)
print(testmMat.shape)
column = testmMat.shape[1]
row = testmMat.shape[0]

result = np.zeros((row,column,3))
for j in range(column-1):
    if j==0 or j==column-2:
        a=1

    else:
        for d in range(3):
            result[:,j,d] = 0.1 * (2 * testmMat[:,j,d] - testmMat[:,j+1,d] - testmMat[:,j-1,d])

imsave('image/testResult.png',result)
