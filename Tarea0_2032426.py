import numpy as np
from math import exp, log, sin, cos
from random import random
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt


data = [11, 48, 23, 48, 25, 56, 16, 76, 48, 76, 13, 12, 25, 36, 79]
anotherData = [1, 8, 3, 4, 5, 6, 1, 76, 4, 16, 13, 12, 5, 3, 79]


if data == anotherData:
     print("Las listas son iguales")
else:
     print("No son iguales las listas")

np.mean(data)
np.unique(data, return_counts=True)
np.var(data)
np.std(data)
np.corrcoef(data, anotherData)[0,1]

c = [x == y for (x, y) in zip(data, anotherData)]

m = np.matrix([data, anotherData])


log(data[1])
sin(data[6])
rand(5)

plt.plot(data, anotherData)
plt.hist(data)