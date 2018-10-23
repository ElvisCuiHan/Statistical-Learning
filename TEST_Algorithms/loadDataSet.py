import numpy as np
#from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.misc import *

def regressionDataSet():
    x  = list(np.hstack((np.linspace(0,15,100), np.linspace(0.01,15,50))).T)
    x = np.array(sorted(x))

    y1 = np.linspace(0,10,100) + np.random.random(100)
    y2 = np.linspace(0,10,50)  + np.random.random(50)
    y3 = 15 * (0.2 * np.sin(0.5 * x) + 0.3 * np.cos(0.6 * x) + 0.22 * np.random.random(150) )
    y4 = 0.002 * x**3 + x**2 -9*x +1 + 20 * np.random.random(150)
    y  = np.hstack((y1, y2)).T
    plt.scatter(x,y3,c='b',s=5)
    plt.savefig('image/originalRegressionImage.png')
    return x, y3

def gaussianDataSet():
    mean1 = np.array([5,1]); mean2 = np.array([1,5])
    cov1 = np.array([[5,0],[0,1]]); cov2 = np.array([[1,0],[0,5]])
    dataMat = np.vstack((np.random.multivariate_normal(mean1,cov1,100), \
                         np.random.multivariate_normal(mean2, cov2, 100)))
    labelMat = -np.ones(200)
    labelMat[101:] = 1
    plt.scatter(dataMat[:100,0],dataMat[:100,1],c='b',s=1)
    plt.scatter(dataMat[101:, 0], dataMat[101:, 1], c='g',s=1)
    plt.savefig('image/originalGaussianDataSet.png')
    return dataMat[:], labelMat

def fisherDataSet():
    mean1 = np.array([4,1]); mean2 = np.array([1,2])
    cov1 = np.array([[1.5,0.8],[0.8,1]]); cov2 = np.array([[1.5,0.8],[0.8,1]])
    dataMat = np.vstack((np.random.multivariate_normal(mean1,cov1,100), \
                         np.random.multivariate_normal(mean2, cov2, 100)))
    labelMat = -np.ones(200)
    labelMat[101:] = 1
    plt.scatter(dataMat[:100,0],dataMat[:100,1],c='b',s=1)
    plt.scatter(dataMat[101:, 0], dataMat[101:, 1], c='g',s=1)
    plt.savefig('image/originalFisherDataSet.png')
    return dataMat[:], labelMat

def ringDataSet():
    x = 0.01 * np.random.randint(-400,400,100)
    y1 = np.sqrt(16 - x[:50]**2) + 1 * np.random.random(50) - 2
    y2 = -np.sqrt(16 - x[50:]**2) + 1 * np.random.random(50) + 0.5
    y = np.vstack((y1,y2))
    x = x.reshape(100,1)
    y = y.reshape(100,1)
    dataMat = np.hstack((x,y))
    print(dataMat.shape)
    mean = [0,0]
    cov = [[3,0],[0,1]]
    dataMat = np.vstack((dataMat, np.random.multivariate_normal(mean, cov, 100)))

    labelMat = np.append(np.zeros(100),np.ones(100))
    plt.scatter(dataMat[:100, 0],dataMat[:100,1],c='b')
    plt.scatter(dataMat[100:, 0], dataMat[100:, 1], c='r')
    plt.savefig('image/originalRingDataSet.png')
    return dataMat, labelMat

def probDeterModel():
    mean1 = np.array([-2,-2])
    mean2 = np.array([2,2])
    cov1 = np.array([[0.5, 0], [0, 0.05]])
    cov2 = np.array([[0.05, 0], [0, 0.05]])
    dataMat = np.vstack((np.random.multivariate_normal(mean1, cov1, 50), \
                         np.random.multivariate_normal(mean2, cov2, 50)))
    mean3 = [0,0]
    cov3 = [[0.5, 0], [0, 0.5]]
    dataMat = np.vstack((dataMat, np.random.multivariate_normal(mean3, cov3, 50)))
    labelMat = np.zeros((150,1))
    labelMat[101:] = 1

    plt.scatter(dataMat[:100,0],dataMat[:100,1],c='b',s=10)
    plt.scatter(dataMat[101:,0],dataMat[101:,1],c='r',s=10)
    plt.savefig('image/originalProbDeterModel.png')

    return dataMat, labelMat

def singleVariableRegressionDataSet():
    data = np.linspace(0, 50, 100)
    y = np.sin(data[:50])
    data = (data - np.mean(data)) / np.var(data)
    y = (y - np.mean(y)) / np.var(y)
    #plt.plot(x[:50],y)
    #plt.show()
    return data, y