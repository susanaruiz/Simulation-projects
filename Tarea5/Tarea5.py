""" Método Monte-Carlo, hallar la aproximación de un valor que 
resulta complicado de determinar de manera analítica. """  

from math import exp, pi
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def g(x):
    return (2  / (pi * (exp(x) + exp(-x))))
 
vg = np.vectorize(g)
X = np.arange(-8, 8, 0.05) 
Y = vg(X) 
 
from GeneralRandom import GeneralRandom
generador = GeneralRandom(np.asarray(X), np.asarray(Y))
muestra = [10000, 50000, 100000, 500000, 1000000]
for t in muestra:
    desde = 3
    hasta = 7
    pedazo = t
    iteraciones = 10
    cuantos = 100
def parte(replica):
    V = generador.random(pedazo)[0]
    return ((V >= desde) & (V <= hasta)).sum() 
datos = []
datos10000 = []
datos50000 = []
datos100000 = []
datos500000 = []
datos1000000 = []
alpha = 0.0488341
integral = []
import multiprocessing
if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        for i in range(iteraciones):
            montecarlo = pool.map(parte, range(cuantos))
            integral.append((pi / 2) * sum(montecarlo) / (cuantos * t))
            error = ((alpha - integral) / alpha)

        datos.append()
        #print(datos)
        if t==10000:
            datos10000=datos
        elif t==50000:
            datos50000=datos
        elif t==100000:
            datos100000=datos
        elif t==500000:
            datos500000=datos
        elif t==1000000:
            datos1000000=datos
        
        # Figuras
        df = pd.read_csv("errorc.csv")
        df = df[['x', 'e10000']]
        x=df['x']
        y=df['e10000']
        plt.figure(figsize=(7,7))
        ax = sns.boxplot(x=x, y=y, data=df)
        df.boxplot(by ='x', column =['e10000'], grid = False)
        plt.boxplot(x, y)
        plt.show

        df = pd.read_csv("errord.csv")
        df = df[['x', 'e50000']]
        x=df['x']
        y=df['e50000']
        plt.figure(figsize=(7,7))
        ax = sns.boxplot(x=x, y=y, data=df)
        df.boxplot(by ='x', column =['e50000'], grid = False)
        plt.boxplot(x, y)
        plt.show
        
        df = pd.read_csv("errore.csv")
        df = df[['x', 'e100000']]
        x=df['x']
        y=df['e100000']
        plt.figure(figsize=(7,7))
        ax = sns.boxplot(x=x, y=y, data=df)
        df.boxplot(by ='x', column =['e100000'], grid = False)
        plt.boxplot(x, y)
        plt.show

        df = pd.read_csv("errorf.csv")
        df = df[['x', 'e500000']]
        x=df['x']
        y=df['e500000']
        plt.figure(figsize=(7,7))
        ax = sns.boxplot(x=x, y=y, data=df)
        df.boxplot(by ='x', column =['e500000'], grid = False)
        plt.boxplot(x, y)
        plt.show


        df = pd.read_csv("errorg.csv")
        df = df[['x', 'e1000000']]
        x=df['x']
        y=df['e1000000']
        plt.figure(figsize=(7,7))
        ax = sns.boxplot(x=x, y=y, data=df)
        df.boxplot(by ='x', column =['e1000000'], grid = False)
        plt.boxplot(x, y)
        plt.show

