""" Modelo de urnas: fenómenos de coalescencia y fragmentación """

import numpy as np
from random import randint
 
k = [100, 250, 500, 1000]
n = [10000, 50000, 75000, 100000]
t = [10, 25, 50, 100]
 
from math import exp, floor, log
 
def rotura(x, c, d):
    return 1 / (1 + exp((c - x) / d))
 
def union(x, c):
    return exp(-x / c)
 
from random import random
 
def romperse(tam, cuantos, c, d):
    if tam == 1: # no se puede romper
        return [tam] * cuantos
    res = []
    for cumulo in range(cuantos):
        if random() < rotura(tam, c, d):
            primera = randint(1, tam - 1)
            segunda = tam - primera
            assert primera > 0
            assert segunda > 0
            assert primera + segunda == tam
            res += [primera, segunda]
        else:
            res.append(tam) # no rompió
    assert sum(res) == tam * cuantos
    return res
 
def unirse(tam, cuantos, c):
    res = []
    for cumulo in range(cuantos):
        if random() < union(tam, c):
            res.append(-tam) # marcamos con negativo los que quieren unirse
        else:
            res.append(tam)
    return res
 
from numpy.random import shuffle
import matplotlib.pyplot as plt
 
duracion = 10
digitos = floor(log(duracion, 10)) + 1
def filtrar(k, n, t):
    orig = np.random.normal(size = k)
    cumulos = orig - min(orig)
    cumulos += 1 # ahora el menor vale uno
    cumulos = cumulos / sum(cumulos) # ahora suman a uno
    cumulos *= n # ahora suman a n, pero son valores decimales
    cumulos = np.round(cumulos).astype(int) # ahora son enteros
    diferencia = n - sum(cumulos) # por cuanto le hemos fallado
    cambio = 1 if diferencia > 0 else -1
    while diferencia != 0:
        p = randint(0, k - 1)
        if cambio > 0 or (cambio < 0 and cumulos[p] > 0): # sin vaciar
            cumulos[p] += cambio
            diferencia -= cambio
    assert all(cumulos != 0)
    assert sum(cumulos) == n
    c = np.median(cumulos) # tamaño crítico de cúmulos
    d = np.std(cumulos) / 4 # factor arbitrario para suavizar la curva
    for paso in range(t):
        assert all([c > 0 for c in cumulos]) 
        (tams, freqs) = np.unique(cumulos, return_counts = True)
        cumulos = []
        assert len(tams) == len(freqs)
        for i in range(len(tams)):
            cumulos += romperse(tams[i], freqs[i], c, d) 
        assert sum(cumulos) == n
        assert all([c > 0 for c in cumulos]) 
        (tams, freqs) = np.unique(cumulos, return_counts = True)
        cumulos = []
        assert len(tams) == len(freqs)
        for i in range(len(tams)):
            cumulos += unirse(tams[i], freqs[i], c)
        cumulos = np.asarray(cumulos)
        neg = cumulos < 0
        a = len(cumulos)
        juntarse = -1 * np.extract(neg, cumulos) # sacarlos y hacerlos positivos
        cumulos = np.extract(~neg, cumulos).tolist() # los demás van en una lista
        assert a == len(juntarse) + len(cumulos)
        nt = len(juntarse)
        if nt > 1:
            shuffle(juntarse) # orden aleatorio
        j = juntarse.tolist()
        while len(j) > 1: # agregamos los pares formados
            cumulos.append(j.pop(0) + j.pop(0))
        if len(j) > 0: # impar
            cumulos.append(j.pop(0)) # el ultimo no alcanzó pareja
        assert len(j) == 0
        assert all([c != 0 for c in cumulos])
        #cortes = np.arange(min(cumulos), max(cumulos), 50)
    #plt.hist(cumulos, bins = cortes, align = 'right', density = True)
    #plt.xlabel('Tamaño')
    #plt.ylabel('Frecuencia relativa')
    #plt.ylim(0, 0.05)
    #plt.title('Paso {:d} con ambos fenómenos'.format(paso + 1))
    #plt.savefig('p8p_ct' + format(paso, '0{:d}'.format(digitos)) + '.png')
    #plt.close()
        
        # Eliminar los que sobrepasen el valor c : filtrado
            filtrados = []
            noFiltrados = []
            cumTotal = len(cumulos)
            for i in range(len(cumulos)):
                c = np.median(cumulos)
                if cumulos[i] > c:
                    filtrados.append(cumulos[i])
                else:
                    noFiltrados.append(cumulos[i])
            #cumulos = noFiltrados
    return ((len(filtrados) / cumTotal)*100)

dataSafe3 = []
for cantidad in k:
    dataSafe3 = []
    for particulas in n:
        dataSafe2 = []
        for replicas in t:
            dataSafe1 = []
            for d in range(duracion):
                res = filtrar(cantidad, particulas, replicas)
        dataSafe3.append(dataSafe2)

    # Figuras

    plt.subplot(223)
    box = plt.boxplot(dataSafe3[2], notch=True, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'tan', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks([1, 2, 3, 4], t)
    plt.title('75000')

    plt.subplot(224)
    box = plt.boxplot(dataSafe3[3], notch=True, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'tan', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks([1, 2, 3, 4], t)
    plt.title('100000')
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.2)
    plt.savefig('t8fig' + format(cantidad, '0{:d}'.format(digitos)) + '.png')
    plt.close()