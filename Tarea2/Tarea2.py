"""El juego de la vida. Se tiene una matriz de 20x20 con celdas 
de valores de 0 y 1, en cada iteración mueren o se quedan vivas dichas
celdas, dependiendo de si el número de vecinos es igual a 3"""

from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

# Variables 
dimention = 20
orden = dimention**2
probabilidadVivir = 0.1

values = [0] * orden
for i in range(orden):
    generacionNumeros = random()
    if generacionNumeros < probabilidadVivir:
        values[i] = 1
    else:
        values[i] = 0
matriz = np.reshape(values, (dimention, dimention))

def gameOfLife(position):
    fila = position // dimention
    columna = position % dimention
    vecindad = matriz[(max(0, fila-1)):min(dimention, fila+2), (max(0, columna-1)):min(dimention, columna+2)]
    return 1 * (np.sum(vecindad) - matriz[fila, columna] == 3)
print(matriz)
vivos = sum(values)
print("Iter 0")
print(0, vivos)
# Imágenes para GIF
fig = plt.figure()
plt.imshow(matriz, interpolation='none', cmap=cm.Greys)
fig.suptitle('Iteración 0')
plt.savefig('p2_i0_p.png')
plt.close()

vivosActual = 0
colapso = 0
colapsoMayor = 0
for iteracion in range(49):
    values = [gameOfLife(x) for x in range(orden)]
    colapso = vivosActual - vivos
    vivosActual = vivos
    if colapso > colapsoMayor:
        colapsoMayor = colapso
    vivos = sum(values)
    if vivos == 0:
        print('No queda ninguno vivo.')
        break
    matriz = np.reshape(values, (dimention, dimention))
    print(matriz)
    print("Iter", iteracion + 1)
    print(iteracion + 1, vivos)
    print(colapso)
    # Imágenes para GIF
    fig = plt.figure()
    plt.imshow(matriz, interpolation='none', cmap=cm.Greys)
    fig.suptitle('Iteración {:d}'.format(iteracion + 1))
    plt.savefig('p2_i{:d}_p.png'.format(iteracion + 1))
    plt.close()

# Gráfica de cantidad de vivos por Probabilidad
df = pd.read_csv("colapso.csv", sep=",")
sns.pairplot(df)
# Gráfica de cantidad de vivos por iteraciones 
df.plot(kind='bar',stacked=False,figsize=(20,6),title="Máximo de vivos por iteraciones")
plt.ylabel("Máximo de vivos", fontsize=14, labelpad=15)
plt.xlabel("Iteraciones", fontsize=14, labelpad=15)
plt.show