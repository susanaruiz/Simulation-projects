""" Interacciones entre partículas """

import numpy as np
import pandas as pd
 
n = 50
fuerzaG = 6.67*10**(-11)  # Se introduce una fuerza gravitacional
m = np.random.normal(size = n) # Se agrega la masa
x = np.random.normal(size = n)
y = np.random.normal(size = n)
c = np.random.normal(size = n)
xmax = max(x)
xmin = min(x)
x = (x - xmin) / (xmax - xmin) # de 0 a 1
ymax = max(y)
ymin = min(y)
y = (y - ymin) / (ymax - ymin) 
cmax = max(c)
cmin = min(c)
c = 2 * (c - cmin) / (cmax - cmin) - 1 # entre -1 y 1
g = np.round(5 * c).astype(int)
mmax = max(m)
mmin = min(m)
m = 6 * (m - mmin) / (mmax - mmin) - 1 # entre 10 y 50
m = np.round(m).astype(int)
m = abs(m*10)
p = pd.DataFrame({'x': x, 'y': y, 'c': c, 'g': g, 'm': m})
paso = 256 // 10
niveles = [i/256 for i in range(0, 256, paso)]
colores = [(niveles[i], 0, niveles[-(i + 1)]) for i in range(len(niveles))]
 
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
from matplotlib.colors import LinearSegmentedColormap

palette = LinearSegmentedColormap.from_list('tonos', colores, N = len(colores))

from math import fabs, sqrt, floor, log
eps = 0.001
def fuerza(i):
    pi = p.iloc[i]
    xi = pi.x
    yi = pi.y
    ci = pi.c
    fx, fy = 0, 0
    for j in range(n):
        pj = p.iloc[j]
        cj = pj.c
        xj = pj.x
        yj = pj.y
        dire = (-1)**(1 + (ci * cj < 0))
        dx = xi - pj.x
        dy = yi - pj.y
        factor = dire * fabs(ci - cj) / (sqrt(dx**2 + dy**2) + eps)
        fx -= dx * factor
        fy -= dy * factor
    return (fx, fy)

def fuerzaGrav(i):
    pi = p.iloc[i]
    xi = pi.x
    yi = pi.y
    mi = pi.m
    fgx, fgy = 0, 0
    for j in range(n):
        pj = p.iloc[j]
        mj = pj.m
        xj = pj.x
        yj = pj.y
        dx = xi - pj.x
        dy = yi - pj.y
        factor = (fuerzaG * ((mi - mj) / (sqrt((dx**2) + (dy**2)) + eps)**2))
        fgx -= dx * factor
        fgy -= dy * factor
    return (fgx*10000000, fgy*10000000)

#from os import popen
#popen('rm -f p9p_t*.png') # borramos anteriores en el caso que lo hayamos corrido
tmax = 50
digitos = floor(log(tmax, 10)) + 1
fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
pos = plt.scatter(p.x, p.y, c = p.g, s = m, cmap = palette)
fig.colorbar(pos, ax=ax)
plt.title('Estado inicial')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
fig.savefig('p9p_t0.png')
plt.close()
 
def actualiza(pos, fuerza, de):
    return max(min(pos + de * fuerza, 1), 0)
    
def agrupar(f, fg):
    grupx = []
    grupy = []
    for v in range(n):
        Fx = f[v][0]
        Fgx = fg[x][0] 
        grupx.append(Fx + Fgx)

        Fy = f[v][1]
        Fgy = fg[x][1]
        grupy.append(Fy + Fgy)
    return list(zip(grupx, grupy))


#import multiprocessing
#from itertools import repeat
 
if __name__ == "__main__":
    for t in range(tmax):
        #with multiprocessing.Pool() as pool: # rehacer para que vea cambios en p
        valores = []
        valores2 =[]
        f = []
        fg = []
        xactual = []
        yactual = []
        q = 0
        Q = 0
        for i in range(n):    
            f.append(fuerza(i))
            fg.append(fuerzaGrav(i))

        F = agrupar(f, fg)
        delta = 0.02 / max([max(fabs(fx), fabs(fy)) for (fx, fy) in F])
        for v in F:
            xactual.append(actualiza(p.x[q], v[0], delta))
            q = q + 1
        p['x'] = xactual
        for v in F:
            yactual.append(actualiza(p.y[Q], v[1], delta))
            Q = Q + 1
        p['y'] = yactual
        
        #Figuras
        fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
        pos = plt.scatter(p.x, p.y, c = p.g, s = m, cmap = palette)
        fig.colorbar(pos, ax=ax)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)            
        plt.title('Paso {:d}'.format(t + 1))
        fig.savefig('p9p_t' + format(t + 1, '0{:d}'.format(digitos)) + '.png')
        plt.close()

        plt.plot(range(tmax), valores, label = 'Masa agregada' )
        plt.plot(range(tmax), valores2, label = 'Sin masa' )
        plt.xlabel('Iteración')
        plt.ylabel('Velocidad')
        plt.legend()
        plt.show()
        plt.close()
