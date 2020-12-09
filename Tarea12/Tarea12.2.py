""" Red neuronal"""

from random import randint
from math import floor, log
import pandas as pd
import numpy as np
 
modelos = pd.read_csv('digits.txt', sep=' ', header = None)
#modelos = modelos.replace({'n': 0.995, 'g': 0.92, 'b': 0.002})
modelos = modelos.replace({'n': 0.605, 'g': 0.82, 'b': 0.992})
#modelos = modelos.replace({'n': 0.902, 'g': 0.99 , 'b': 0.005})
r, c = 5, 3
dim = r * c 
 
tasa = 0.15
tranqui = 0.99
tope = 9
k = tope + 1 # incl. cero
contadores = np.zeros((k, k + 1), dtype = int)
n = floor(log(k-1, 2)) + 1
neuronas = np.random.rand(n, dim) # perceptrones
  
for t in range(5000): # entrenamiento
    d = randint(0, tope)
    pixeles = 1 * (np.random.rand(dim) < modelos.iloc[d])
    correcto = '{0:04b}'.format(d)
    for i in range(n):
        w = neuronas[i, :]
        deseada = int(correcto[i]) # 0 o 1
        resultado = sum(w * pixeles) >= 0
        if deseada != resultado: 
            ajuste = tasa * (1 * deseada - 1 * resultado)
            tasa = tranqui * tasa 
            neuronas[i, :] = w + ajuste * pixeles
 
for t in range(300): # prueba
    d = randint(0, tope)
    pixeles = 1 * (np.random.rand(dim) < modelos.iloc[d])
    correcto = '{0:04b}'.format(d)
    salida = ''
    for i in range(n):
        salida += '1' if sum(neuronas[i, :] * pixeles) >= 0 else '0'
    r = min(int(salida, 2), k)
    contadores[d, r] += 1
c = pd.DataFrame(contadores)
c.columns = [str(i) for i in range(k)] + ['NA']
c.index = [str(i) for i in range(k)]
print(c)
#print(contadores)
from sklearn import metrics

# Constants
A="0"
B="1"
C="2"
D="3"
E="4"
F="5"
G="6"
H="7"
I="8"
J="9"
# True values
y_true = [A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A,A, B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B, C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C, D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D, E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E, F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F, G,G,G,G,G,G,G,G,G,G,G,G,G,G,G,G, H,H,H,H,H,H,H,H,H,H,H,H,H, I,I,I, J,J,J,J,J,J,J,J,J,J,J,J,J,J]
# Predicted values
y_pred = [A,A,A,B,B,B,B,B,B,C,C,C,D,D,D,D,D,D,E,E,E,F,F,F,F,F,G,G,G,G,H,H,I,I,I,J, A,A,A,A,A,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,C,C,C,C,D,D,D,D,D,D,D,D,E,E,E,E,E,F,F,G,H,J, A,A,A,A,A,A,C,C,C,C,C,C,C,C,C,D,D,E,E,E,F,F,F,C,H,C,I,I,I,I,I,I,I,I,J, A,C,C,C,C,C,C,D,D,D,D,E,E,F,G,H,I,I,I,J, A,A,A,D,D,D,D,E,E,F,G,G,H,H,I,I,I,I,J,J,E,E,E,E,E,E,E,E,E,F,F,F,F,F,F,F,F,G,I,J,J, D,D,D,E,E,E,G,G,F,F,F,F,F,F,F,F,F,F,F,H,H,H,H,H,H,H,H, A,C,C,C,C,C,D,D,D,D,D,F,F,F,I,G, H,H,H,H,E,E,F,F,F,H,H,H,J, G,G,G, A,E,E,E,E,F,F,F,F,F,F,F,G,J]

# Print the confusion matrix
#print(metrics.confusion_matrix(y_true, y_pred))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))

import matplotlib.pyplot as plt
# Ordenamiento
data1 = [0.700,0.999,0.999,0.971,0.999,0.999,0.731,0.950,0.406,0.784]
data2 = [0.929,0.914,0.824,0.650,0.800,0.519,0.722,0.718,0.625,0.480]
data3 = [0.150,0.739,0.379,0.125,0.333,0.250,0.062,0.318,0.000,0.100]


fig = plt.figure(figsize=(8,6))

plt.boxplot([x for x in [data1, data2, data3]], 0, 'rs', 1)
plt.xticks([y+1 for y in range(len([data1, data2, data3]))], ['Probabilidad1', 'Probabilidad2', 'Probabilidad3'])
plt.ylabel('Porcentaje de precisión')
t = plt.title('Gráfica de precisión variando las probabilidades')
plt.show()