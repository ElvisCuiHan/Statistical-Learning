import numpy as np
from decimal import *
sum = 0

for n in range(1000):
    an = n * Decimal(np.power(1/3.0,n-1))
    sum += an

print(sum)