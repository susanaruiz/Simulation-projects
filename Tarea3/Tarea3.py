""" Teoría de colas """   

import numpy as np
from time import time
from sys import getsizeof
from math import ceil, sqrt
from scipy.stats import describe
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
from random import shuffle
import psutil
from multiprocessing import Pool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

# Importar números primos y crear números fáciles
with open('datosprim.txt', 'r') as input:
  read = input.readline()
primos = [int(valor) for valor in read.split(',')]
cantidadPrimos = len(primos)


minimo = 982451629      # Máximos números primos
maximo = 982451653
facil = []
for cand in range(minimo+1, maximo-1):
  facil.append(cand)
total = primos + facil
cantidadTotal = len(total)

def numerosPrimos(n):
  if n < 4:
    return True
  if n % 2 == 0:
    return False
  for i in range(3, int(ceil(sqrt(n))), 2):
    if n % i == 0:
      return False
  return True

nucleo = psutil.cpu_count()
from multiprocessing import Pool
if __name__ == "__main__":
  original = total
  invertido = original[::-1]
  aleatorio = original.copy()
  replicas = 10
  tiempos = {"or": [], "in": [], "al": []}
  for n in range(1, nucleo+1):
  with Pool(n) as p:
    for r in range(replicas):
      t = time()
      p.map(numerosPrimos, original)
      tiempos["or"].append(time() - t)
      t = time()
      p.map(numerosPrimos, invertido)
      tiempos["in"].append(time() - t)
      shuffle(aleatorio)
      t = time()
      p.map(numerosPrimos, aleatorio)
      tiempos["al"].append(time() - t)
  for tipo in tiempos:
    describe(tiempos[tipo])

#Pruebas Estadísticas

# Prueba Student's t-test (Comprueba si las medias de dos muestras emparejadas son significativamente diferentes)
data1 = [0.32102465629577637, 2.5551886558532715, 0.5686076641082763, 0.4877347171961168]
data2 = [0.32302260398864746, 0.3776113986968994, 0.3506206512451172, 0.00036365630901905705]
stat, p = ttest_rel(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# Prueba de correlación de Pearson (Prueba si dos muestras tienen una relación lineal) 
data1 = [0.32102465629577637, 2.5551886558532715, 0.5686076641082763, 0.4877347171961168]
data2 = [0.32302260398864746, 0.3776113986968994, 0.3506206512451172, 0.00036365630901905705]
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#Gráficas

fig = plt.figure(figsize=(8,6))

plt.boxplot([x for x in [x1, x2, x3, x4]], 0, 'rs', 1)
plt.xticks([y+1 for y in range(len([x1, x2, x3, x4]))], ['1', '2', '3', '4'])
plt.xlabel('Nucleos')
t = plt.title('Box plot')
plt.show()

# Ordenamiento

fig = plt.figure(figsize=(8,6))

plt.boxplot([x for x in [data1, data2, data3]], 0, 'rs', 1)
plt.xticks([y+1 for y in range(len([data1, data2, data3]))], ['Normal', 'Inverso', 'Aleatorio'])
plt.xlabel('Por Orden')
t = plt.title('Box plot')
plt.show()

plt.plot(data1, marker='x', linestyle=':', color='b', label = "PrimosPrimero")
plt.plot(data2, marker='*', linestyle='-', color='g', label = "NoPrimosPrimero")
plt.plot(data3, marker='o', linestyle='--', color='r', label = "Aleatorio")
plt.xlabel('Diferencias de ejecución de los ordenamientos')
plt.legend(loc="upper right")
plt.show()