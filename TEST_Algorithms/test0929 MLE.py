import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from TEST_Algorithms.loadDataSet import singleVariableRegressionDataSet

#x = [2.17,4.5,13.33,24.5,29.67,35,49.67,65.5,81.33]
#t = [0.04,0.039,0.038,0.024,0.021,0.023,0.017,0.017,0.013]

data, t = singleVariableRegressionDataSet()

t = t/10
x2 = data[50:]
x = data[:50]

plt.scatter(data[:50],t,s=40)
#x,t = regressionDataSet()
pinv = np.linalg.pinv

basis_num = 10
sigma = 0.01
phi_mu = np.linspace(min(x),max(x),basis_num)
print("phi_mu is : ", phi_mu)
phi_sigma = sigma * np.ones(basis_num)

phi_matrix = np.ones((len(data),basis_num))
for j in range(basis_num):
    aaa = stats.norm.pdf(data,phi_mu[j],phi_sigma[j]) #* np.sqrt(2*np.pi) * phi_sigma[j]
    phi_matrix[:,j] = aaa

phi_pred = phi_matrix[:50]
print(phi_matrix[50:])

w_ml = np.dot(np.dot(pinv(np.dot(phi_pred.T,phi_pred)),phi_pred.T),t)

print("w_ml is : ", w_ml)

y_fitting = np.dot(phi_matrix,w_ml)
#y_pred = np.dot(phi_pred,w_ml)
#print(phi_matrix)
plt.plot(data,y_fitting,c='r')

plt.savefig('image/MLE.png')
plt.show()
