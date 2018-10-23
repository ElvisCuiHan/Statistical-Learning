from TEST_Algorithms.loadDataSet import *

pinv = np.linalg.pinv


dataMat, labelMat = gaussianDataSet()
dataMat = np.hstack((np.ones((200,1)), dataMat))
plt.show()
weightMat = np.dot(np.dot(pinv(np.dot(dataMat.T, dataMat)), dataMat.T), labelMat)
print(weightMat)

x = np.linspace(1,10,10)
y = np.dot(dataMat, weightMat)
num = 0
for i in y:

    if(i>0):
        plt.scatter(dataMat[num,1],dataMat[num,2],c='r',s=5)
    else:
        plt.scatter(dataMat[num,1],dataMat[num,2],c='y',s=5)
    num += 1
plt.savefig('image/leastSquare4Classification.png')
plt.show()