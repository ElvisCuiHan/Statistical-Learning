from TEST_Algorithms.loadDataSet import *
dot = np.dot
pinv = np.linalg.pinv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

dataMat, labelMat = probDeterModel()

mean1 = np.array([0,0])
invCov1 = np.array([[0.5,0],[0,0.5]])
mean2 = np.array([0,0])
invCov2 = np.array([[2,0],[0,2]])

phi = np.ones((150,3))
for i in range(150):
    phi[i,0] = np.exp(-0.5 * dot(dot((dataMat[i,:] - mean1).T, invCov1), (dataMat[i,:] - mean1))) * np.linalg.det(invCov1)
    phi[i,1] = np.exp(-0.5 * dot(dot((dataMat[i,:] - mean2).T, invCov2), (dataMat[i,:] - mean2))) * np.linalg.det(invCov2)

wOld = np.ones((3,1))/2
dataMatNew = np.hstack((np.ones((150,1)),dataMat))
wNew = np.ones((3,1))
y = phi[:,0] - phi[:,1]
print(y)

num = 0
for i in y:
    if i > 0:
        plt.scatter(dataMat[num,0], dataMat[num,1], c='b')
    else :
        plt.scatter(dataMat[num,0], dataMat[num,1], c='r')
    num += 1
plt.savefig('image/PDA.png')
plt.show()

