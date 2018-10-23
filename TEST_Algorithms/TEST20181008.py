import numpy as np
import matplotlib.pyplot as plt

result = []

for i in range(200):
    x1 = np.random.normal(10,3)
    x2 = np.random.normal(20,2)
    y = x1 + x2
    result.append(y)

def emf(x,n):
    prob = 0
    for i in result:
        if(x>=i):
            prob += 1.0/n
        #print(i)
    return prob

for j in range(10,50):
    prob = emf(j,200)
    plt.scatter(j,prob,c='r')
plt.show()
