""" Algoritmo genético: """

import numpy as np
import pandas as pd
from random import random, randint, sample, choices
from time import time
 
def knapsack(peso_permitido, pesos, valores):
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

def fit(fact, obj):
    #fit = [random() - 0.1 for x in n] # selección de ruleta
    if fact != True:
        obj = 0.10 * obj
        return obj

def ruleta(pop, fitness):
    sel = {}
    while len(sel) < 2:
        sel = set(choices(pop, weights=fitness, k=2))
    return list(sel)

def genetico(n, k):
    #n = 50
    pesos = generador_pesos(n, 15, 80)
    valores = generador_valores(pesos, 10, 500)
    capacidad = int(round(sum(pesos) * 0.65))
    valor = 0
    tiempoIni = time()
    optimo = knapsack(capacidad, pesos, valores)
    tiempoFin = time()
    tiempTot = tiempoFin - tiempoIni
    datos = optimo / tiempTot
    init = 40
    p = poblacion_inicial(n, init)
    tam = p.shape[0]
    assert tam == init
    pm = 0.05
    rep = 5
    tmax = 80
    mejor = None
    mejores = []

    c = 0
    tinic = []
    tiempo1 = time()
    for t in range(tmax):
        c = c+1
        time1 = time()
        if k == 0:
            fitnes = []
            valobj = []
            fac = []
            tam = p.shape[0]
            for i in range(tam):
                valobj.append(objetivo(p[i], valores))
                fac.append(factible(p[i], pesos, capacidad))
            valorfit = list(zip(valobj, fac))

            for i in range(len(valorfit)):
                fitnes.append(fit(valorfit[i][1], valorfit[i[0]]))

            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(valobj)), fitnes)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
        if k == 1:
            for i in range(tam): 
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
    tam = p.shape[0]
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
    
    time2 = time()
    tinic.append(time2 - time1)
    timet = sum(tinic)
    datos2 = mejor / timet

    if timet >= tiempTot:
        break
    if datos < datos2:
        

# Regla 1: peso y valor distribución exponencial
def genetico1(n, k):
    #n = 50
    pesos = generador_pesos(n, 15, 80)
    valores = generador_valores(pesos, 10, 500)
    capacidad = int(round(sum(pesos) * 0.65))
    valor = 0
    tiempoIni = time()
    optimo = knapsack(capacidad, pesos, valores)
    tiempoFin = time()
    tiempTot = tiempoFin - tiempoIni
    datos = optimo / tiempTot
    init = 40
    p = poblacion_inicial(n, init)
    tam = p.shape[0]
    assert tam == init
    pm = 0.05
    rep = 5
    tmax = 80
    mejor = None
    mejores = []

    c = 0
    tinic = []
    tiempo1 = time()
    for t in range(tmax):
        c = c+1
        time1 = time()
        if k == 0:
            fitnes = []
            valobj = []
            fac = []
            tam = p.shape[0]
            for i in range(tam):
                valobj.append(objetivo(p[i], valores))
                fac.append(factible(p[i], pesos, capacidad))
            valorfit = list(zip(valobj, fac))

            for i in range(len(valorfit)):
                fitnes.append(fit(valorfit[i][1], valorfit[i[0]]))

            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(valobj)), fitnes)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
        if k == 1:
            for i in range(tam): 
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
    tam = p.shape[0]
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
    
    time2 = time()
    tinic.append(time2 - time1)
    timet = sum(tinic)
    datos2 = mejor / timet

    if timet >= tiempTot:
        break
    
# Regla 2: peso dist. exponencial y valor positivamente correlacionado con el peso, con un ruido
def genetico2(n, k):
    #n = 50
    pesos = generador_pesos(n, 15, 80)
    valores = generador_valores(pesos, 10, 500)
    capacidad = int(round(sum(pesos) * 0.65))
    valor = 0
    tiempoIni = time()
    optimo = knapsack(capacidad, pesos, valores)
    tiempoFin = time()
    tiempTot = tiempoFin - tiempoIni
    datos = optimo / tiempTot
    init = 40
    p = poblacion_inicial(n, init)
    tam = p.shape[0]
    assert tam == init
    pm = 0.05
    rep = 5
    tmax = 80
    mejor = None
    mejores = []

    c = 0
    tinic = []
    tiempo1 = time()
    for t in range(tmax):
        c = c+1
        time1 = time()
        if k == 0:
            fitnes = []
            valobj = []
            fac = []
            tam = p.shape[0]
            for i in range(tam):
                valobj.append(objetivo(p[i], valores))
                fac.append(factible(p[i], pesos, capacidad))
            valorfit = list(zip(valobj, fac))

            for i in range(len(valorfit)):
                fitnes.append(fit(valorfit[i][1], valorfit[i[0]]))

            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(valobj)), fitnes)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
        if k == 1:
            for i in range(tam): 
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
    tam = p.shape[0]
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
    
    time2 = time()
    tinic.append(time2 - time1)
    timet = sum(tinic)
    datos2 = mejor / timet

    if timet >= tiempTot:
        break
    
# Regla 3: peso dist. exponencial y valor inversamente correl con peso
def genetico3(n, k):
    #n = 50
    pesos = generador_pesos(n, 15, 80)
    valores = generador_valores(pesos, 10, 500)
    capacidad = int(round(sum(pesos) * 0.65))
    valor = 0
    tiempoIni = time()
    optimo = knapsack(capacidad, pesos, valores)
    tiempoFin = time()
    tiempTot = tiempoFin - tiempoIni
    datos = optimo / tiempTot
    init = 40
    p = poblacion_inicial(n, init)
    tam = p.shape[0]
    assert tam == init
    pm = 0.05
    rep = 5
    tmax = 80
    mejor = None
    mejores = []

    c = 0
    tinic = []
    tiempo1 = time()
    for t in range(tmax):
        c = c+1
        time1 = time()
        if k == 0:
            fitnes = []
            valobj = []
            fac = []
            tam = p.shape[0]
            for i in range(tam):
                valobj.append(objetivo(p[i], valores))
                fac.append(factible(p[i], pesos, capacidad))
            valorfit = list(zip(valobj, fac))

            for i in range(len(valorfit)):
                fitnes.append(fit(valorfit[i][1], valorfit[i[0]]))

            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(valobj)), fitnes)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
        if k == 1:
            for i in range(tam): 
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
    tam = p.shape[0]
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
    
    time2 = time()
    tinic.append(time2 - time1)
    timet = sum(tinic)
    datos2 = mejor / timet

    if timet >= tiempTot:
        break
values = []
for k in range(2):
    for l in range(4):
        for i in range(40):
            for n in range(10,50):
                if l == 0:
                    exp = []
                    exp = genetico(n,k)
                    if exp != None:
                        if exp[0] == 999:
                            if k == 0:
                                values.append(exp[1])
                                break
                            if k == 1:
                                values.append(exp[1])
                                break    

import matplotlib.pyplot as plt
  
pplt.figure(figsize=(7, 3))
plt.axhline(y = optimo2, color = 'green')
plt.plot(range(tmax), mejores,linewidth=1, markersize=3, label = 'Sin ruleta', color = 'black')
plt.plot(range(tmax), mejores2, linewidth=1, markersize=3, label = 'Con ruleta', color = 'orange')
plt.xlabel('Iteraciones')
plt.legend()
plt.ylim(0.95 * min(mejores2), 1.05 * optimo2)
plt.show()
plt.close()
#print(mejor2, (optimo2 - mejor2) / optimo2)
#print(mejores)
#print(mejores2)

# Pruebas Estadísticas
from scipy.stats import f_oneway
data1 = [8230.745122711312, 8230.745122711312, 8280.212340997594, 8378.434874254403, 8378.434874254403, 8378.565546938236, 8391.196828942282, 8409.38058909023, 8412.313054963128, 8412.313054963128, 8412.313054963128, 8412.313054963128, 8422.38270606136, 8427.765237711488, 8433.342293925078, 8433.342293925078, 8433.342293925078, 8502.093450581044, 8502.093450581044, 8502.093450581044, 8502.093450581044, 8502.093450581044, 8502.093450581044, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847, 8511.028127938847]
data2 = [8197.94669871014, 8197.94669871014, 8224.296952275423, 8295.677672724683, 8324.86739212277, 8375.030069129578, 8375.030069129578, 8375.030069129578, 8397.654752093567, 8397.654752093567, 8397.654752093567, 8416.647106919614, 8416.647106919614, 8416.647106919614, 8416.647106919614, 8416.647106919614, 8416.647106919614, 8416.647106919614, 8416.64710696919614, 8416.647106919614, 8416.647106919614, 8416.647106919614, 8416.647106919614, 69.77174227127, 8469.77174227127, 8469.77174227127, 8469.77174227127, 8469.77174227127, 8474.55038416, 8416.647106919614, 8416.647106919614, 8426.494851965661, 8490.137201291114, 8494.915853542629, 8494.915853542629, 8494.915853542629, 8494.915853542629, 851965661, 8469.77174227127, 8469.77174227127, 8469.77174227127, 8469.77174227127, 8469.77174227127, 8474.550394522785]
stat, p = f_oneway(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')