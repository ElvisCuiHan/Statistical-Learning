import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n=4
cdf = 0
for k in np.linspace(0,4,121):
    pmf =  stats.binom.pmf(k, n, 1 / 6)
    cdf += pmf
    if(k in [0,1,2,3,4]):
        print(k,"pmf",pmf,"cdf",cdf)
    plt.scatter(k,cdf,c="black",s=3)
plt.show()