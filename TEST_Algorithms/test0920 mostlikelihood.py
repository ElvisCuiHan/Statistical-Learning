import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from TEST_Algorithms.loadDataSet import regressionDataSet

x,t = regressionDataSet()
pinv = np.linalg.pinv

basis_num = 5
sigma = 1
phi_mu = np.linspace(0,15,basis_num)
print(phi_mu)
phi_sigma = sigma * np.ones(basis_num)

phi_matrix = np.ones((len(x),basis_num))
phi_pred = np.ones((50,basis_num))
#phi_matrix = np.vstack((phi_matrix,phi_pred))
for j in range(basis_num):
    aaa = stats.norm.pdf(x,phi_mu[j],phi_sigma[j]) #* np.sqrt(2*np.pi) * phi_sigma[j]
    phi_matrix[:,j] = aaa

lamda = 0.5
ide = np.eye(basis_num)
w_ml = np.dot(np.dot(pinv(lamda * ide + np.dot(phi_matrix.T,phi_matrix)),phi_matrix.T),t)

print(w_ml)

y_fitting = np.dot(phi_matrix,w_ml)
#print(phi_matrix)
plt.scatter(x,y_fitting)
plt.savefig('image/MostLikelihood.png')
plt.show()