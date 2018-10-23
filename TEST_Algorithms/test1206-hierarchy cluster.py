from TEST_Algorithms.loadDataSet import *

pinv = np.linalg.pinv

dataMatOrigin, labelMatOrigin = gaussianDataSet()
labelMat1 = np.copy(labelMatOrigin[:15])
dataMat1 = np.copy(dataMatOrigin[:15,:])
disMat1 = 50 * np.ones((len(dataMat1),len(dataMat1)))
print(np.shape(disMat1))
#plt.scatter(dataMatOrigin[:,0],dataMatOrigin[:,1],c='r')
#plt.show()
for iter in range(3):
    flag = 0
    for flag in range(len(dataMat1)):
        for i in range(flag+1,len(dataMat1)):

            dis = np.dot((dataMat1[flag] - dataMat1[i]),(dataMat1[flag] - dataMat1[i]))
            disMat1[i, flag] = dis
    #print(disMat[:,:])

    index = [0,0]
    min = 50
    for j in range(len(disMat1)):
        for i in range(j+1,len(disMat1)):
            #print(disMat[i,j])
            if(disMat1[i,j]<min):
                min = disMat1[i,j]
                index = [i,j]

    dataMat1[index] = np.mean(dataMat1[index],0)
    dataMat1[index[0]]= 50 * np.random.random((1,2))
    labelMat1[index] = 0.5
    print(index)

print(labelMat1)
num = 0
for i in labelMat1:
    if (i==0.5):
        plt.scatter(dataMatOrigin[num,0],dataMatOrigin[num,1],c='r')
    else:
        plt.scatter(dataMatOrigin[num,0],dataMatOrigin[num,1],c='b')
    num += 1

plt.show()

