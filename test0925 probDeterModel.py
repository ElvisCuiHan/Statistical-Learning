from loadDataSet import *
dot = np.dot
pinv = np.linalg.pinv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

dataMat, labelMat = probDeterModel()

mean1 = np.array([0,0])
invCov1 = np.array([[0.005,0],[0,0.5]])
mean2 = np.array([0,0])
invCov2 = np.array([[25,0],[0,29]])

phi = np.ones((150,3))
for i in range(150):
    phi[i,0] = np.exp(-0.5 * dot(dot((dataMat[i,:] - mean1).T, invCov1), (dataMat[i,:] - mean1)))
    phi[i,1] = np.exp(-0.5 * dot(dot((dataMat[i,:] - mean2).T, invCov2), (dataMat[i,:] - mean2)))

wOld = np.ones((3,1))/2
dataMatNew = np.hstack((np.ones((150,1)),dataMat))
wNew = np.ones((3,1))
y = np.ones((150,1))
for i in range(100):
    y = sigmoid(dot(phi, wOld))

    #dEw = dot(phi.T,(y - labelMat))
    R = np.zeros((len(y),len(y)))
    #print(R.shape)
    for i in range(len(y)):
        R[i,i] = y[i] * (1 - y[i])
    invH = pinv(dot(phi.T, dot(R, phi)))

    z = dot(phi, wOld) - dot(pinv(R), (y - labelMat))

    wNew = dot(dot(dot(invH, phi.T), R), z)
    wOld = wNew
    #print (wNew)
#print(wNew[1])
#print y
num = 0
for i in y:
    if i > 0.5:
        plt.scatter(dataMat[num,0], dataMat[num,1], c='b')
    else :
        plt.scatter(dataMat[num,0], dataMat[num,1], c='r')
    num += 1
plt.savefig('image/probDeterModel.png')
plt.show()

