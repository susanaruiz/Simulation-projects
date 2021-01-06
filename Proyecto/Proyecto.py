""" Algoritmo genetico: Este proyecto trata de la creación de instancias simuladas con características
similares a datos de instancias verdaderas de objetos para empaquetamiento """ 

import numpy as np
import pandas as pd
from time import time
from random import choices
from random import random, randint, sample

def knapsack(peso_permitido, pesos, valores):    #codigo basado en https://github.com/satuelisa/Simulation/blob/master/GeneticAlgorithm/knapsack.py
    assert len(pesos) == len(valores)
    peso_total = sum(pesos)
    valor_total = sum(valores)
    if peso_total < peso_permitido:
        return valor_total
    else:
        V = dict()
        for w in range(peso_permitido + 1):
            V[(w, 0)] = 0
        for i in range(len(pesos)):
            peso = pesos[i]
            valor = valores[i]
            for w in range(peso_permitido + 1):
                cand = V.get((w - peso, i), -float('inf')) + valor
                V[(w, i + 1)] = max(V[(w, i)], cand)
        return max(V.values())

def factible(seleccion, pesos, capacidad):
    return np.inner(seleccion, pesos) <= capacidad

def objetivo(seleccion, valores):
    return np.inner(seleccion, valores)

def normalizar(data):
    menor = min(data)
    mayor = max(data)
    rango  = mayor - menor
    data = data - menor # > 0
    return data / rango # entre 0 y 1

def generador_pesos(cuantos, low, high):
    return np.round(normalizar(np.random.normal(size = cuantos)) * (high - low) + low)

def generador_valores(pesos, low, high):
    n = len(pesos)
    valores = np.empty((n))
    for i in range(n):
        valores[i] = np.random.normal(pesos[i], random(), 1)
    return normalizar(valores) * (high - low) + low

def poblacion_inicial(n, tam):
    pobl = np.zeros((tam, n))
    for i in range(tam):
        pobl[i] = (np.round(np.random.uniform(size = n))).astype(int)
    return pobl

def mutacion(sol, n):
    pos = randint(0, n - 1)
    mut = np.copy(sol)
    mut[pos] = 1 if sol[pos] == 0 else 0
    return mut

def reproduccion(x, y, n):
    pos = randint(2, n - 2)
    xy = np.concatenate([x[:pos], y[pos:]])
    yx = np.concatenate([y[:pos], x[pos:]])
    return (xy, yx)

def fitness(fact,obj):
    if fact != True:
        obj = obj * .1     # 10% de obj
    return obj

def ruleta(poblacion, fitnes):
    sel = {}
    while len(sel) < 2:
        sel = set(choices(poblacion, weights = fitnes, k = 2))
    return list(sel)

n = 100    #numero de objetos
pesos = generador_pesos(n, 15, 100)
valores = generador_valores(pesos, 10, 900)
capacidad = int(round(sum(pesos) * 0.65))

Toptimo1 = time()
optimo = knapsack(capacidad, pesos, valores)
Toptimo2 = time()
TiempoO = Toptimo2 - Toptimo1
X = optimo / TiempoO

print("Valor optimo:", optimo)
print("El tiempo optimo tardo:",TiempoO,"seg")
print("X:",X,"=", optimo,"/",TiempoO)
print("--------------------------------------")
init = 50  
p = poblacion_inicial(n, init)

tam = p.shape[0]

assert tam == init
pm = 0.05       # probabilidad a mutar
rep = round(n * .2)  
print("rep:",rep)
tmax = 50  
mejor = None
mejores = []

cont = 0
TTin = []
Tres1 = time()
for t in range(tmax):
    cont = cont + 1
    Tin1 = time()
#---------- Generar fit ----------
    fit = []
    fitobj = []
    fitfac = []
    tam = p.shape[0]
    for i in range(tam):
        fitobj.append(objetivo(p[i], valores))
        fitfac.append(factible(p[i], pesos, capacidad))
    FIT = list(zip(fitobj,fitfac))

    #print(FIT)

    for i in range(len(FIT)):
        fit.append(fitness(FIT[i][1],FIT[i][0]))
    

    for i in range(tam): # mutarse con probabilidad pm
        if random() < pm:
            p = np.vstack([p, mutacion(p[i], n)])

    for i in range(rep):  # reproducciones
        padres = ruleta(range(len(fitobj)), fit)
        hijos = reproduccion(p[padres[0]], p[padres[1]], n)
        p = np.vstack([p, hijos[0], hijos[1]])


    tam = p.shape[0]      # tam es el numero de filas en la matriz
    d = []
    for i in range(tam):
        d.append({'idx': i, 'obj': objetivo(p[i], valores),
                  'fact': factible(p[i], pesos, capacidad)})
    d = pd.DataFrame(d).sort_values(by = ['fact', 'obj'], ascending = False)
    mantener = np.array(d.idx[:init])
    p = p[mantener, :]
    tam = p.shape[0]
    assert tam == init
    factibles = d.loc[d.fact == True,]
    mejor = max(factibles.obj)
    mejores.append(mejor)

    Tin2 = time()
    TTin.append(Tin2 - Tin1)
    TTTin = sum(TTin)

    Y = mejor/TTTin
    print(Y,"=", mejor,"/",TTTin)

    if TTTin >= TiempoO:
        print("EN TIEMPO NO FUE MEJOR QUE EL OPTIMO")
        break

    if Y < X :
        print("GENETICO SUPERO EL OPTIMO EN ITERACION:", cont)
        break

    if tmax == cont:
        print("El TIEMPO TERMINO")

Tres2 = time()
import matplotlib.pyplot as plt

print(mejor, (optimo - mejor) / optimo)
print("Y:", mejor / (Tres2 - Tres1))
print(mejor)
print(Tres2 - Tres1)


plt.figure(figsize=(6, 4))
plt.plot(range(cont), mejores,'ks--', linewidth=1, markersize=3)
plt.axhline(y = optimo, color = 'blue')
plt.xlabel('Iteraciones')
plt.ylabel('Función objetivo')
plt.grid(True)
plt.ylim(0.95 * min(mejores), 1.05 * optimo)
plt.show()
plt.close()