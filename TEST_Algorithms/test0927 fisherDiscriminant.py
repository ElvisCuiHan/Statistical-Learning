from TEST_Algorithms.loadDataSet import *
pinv = np.linalg.pinv

dataMat, labelMat = fisherDataSet()

m1 = np.mean(dataMat[:100,:],0)
m2 = np.mean(dataMat[101:,:],0)
print(m1,m2)
Sw = np.array([[.0,.0],[.0,.0]])
for x in dataMat[:100]:
    t = (x - m1).reshape(2,1)
    Sw += t * t.T

for x in dataMat[101:]:
    t = (x - m2).reshape(2,1)
    Sw += t * t.T

pSw = pinv(Sw)

print("Pinv of Sw is: ", pSw)
w = np.dot(pSw, (m2 - m1).T)
print("The weight is:", w)

newData= 50 * np.dot(dataMat, w) + 4
#print(newData)
plt.scatter(newData[:100],np.zeros((100,1)),c='g')
plt.scatter(newData[100:],np.zeros((100,1)),c='b')
plt.savefig('image/fisherDiscriminant.png')