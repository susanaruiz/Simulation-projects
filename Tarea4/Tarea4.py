""" Diagrama de Voronoi. Examinar el tamaño de semillas y la zona 
de la distribución en las grietas en términos de la mayor distancia euclideana"""
import numpy as np
import seaborn as sns 
from random import randint, choice
from PIL import Image, ImageColor 
from math import sqrt
from scipy.stats import describe
import matplotlib.pyplot as plt


size = [60, 80, 100, 150]
tamSemilla = [d for d in np.arange(0.2, 0.8, 0.2)]
tam = [2, 4, 6, 8]
#n = 100
#k = 30
datos = []
semillas = []
def celda(pos):
    y = pos // n
    x = pos % n
    if pos in semillas:
        return semillas.index(pos)
    cercano = None
    menor = n * sqrt(2)
    for i in range(k):
        (xs, ys) = semillas[i]
        dx = x - xs
        dy = y - ys
        dist = sqrt(dx**2 + dy**2)
        if dist < menor:
            cercano = i
            menor = dist
    return cercano


def inicio():
    direccion = randint(0, 3)
    if direccion == 0:
        return (0, randint(0, n - 1))
    elif direccion == 1:
        return (randint(0, n - 1), 0)
    elif direccion == 2:
        return (randint(0, n - 1), n - 1)
    else:
        return (n - 1, randint(0, n - 1))


def propaga(replica):
    probabilidad = 0.9
    dificil = 0.8
    dist = 0
    distEucl = sqrt(((1)**2) + ((1)**2))
    m = n/2
    grieta = voronoi.copy()
    g = grieta.load()
    (x, y) = inicio()
    largo = 0
    negro = (0, 0, 0)
    while True:
        g[x, y] = negro
        largo += 1
        frontera = []
        interior = []
        for v in vecinos:
            (dx, dy) = v
            vx, vy = x + dx, y + dy
            if vx >= 0 and vx < n and vy >= 0 and vy < n:
                if g[vx, vy] != negro:
                    if vor[vx, vy] == vor[x, y]:
                        interior.append(v)
                    else:
                        frontera.append(v)
        elegido = None
        if x < m and y < m:   		# basadoenelcódigodeElitemaster97 https://github.com/Elitemaster97/Simulacion
            xactual, yactual = 0, 0
            dist = sqrt(((x-xactual) **2) + ((y - yactual)**2))
        elif x < m and y > m:
            xactual, yactual = 0, n
            dist = sqrt(((xactual + x) **2) + ((yactual - y)**2))
        elif x > m and y < m:
            xactual, yactual = n, 0
            dist = sqrt(((xactual - x) **2) + ((yactual + y)**2))
        elif x > m and y > m:
            xactual, yactual = n, n
            dist = sqrt(((xactual - x) **2) + ((yactual - y)**2))
        if dist > distEucl:
            distEucl = dist
            distActual = dist

        if len(frontera) > 0:
            elegido = choice(frontera)
            probabilidad = 1
        elif len(interior) > 0:
            elegido = choice(interior)
            probabilidad *= dificil
        if elegido is not None:
            (dx, dy) = elegido
            x, y = x + dx, y + dy
        else:
            break
    return distActual

for n in size:
    datos = []
    for t in tam:
        semillas = []
        k = n * t
        for s in range(k):
            while True:
                x = randint(0, n - 1)
                y = randint(0, n - 1)
                if (x, y) not in semillas:
                    semillas.append((x, y))
                    break
        m = n/2
        distEucl = sqrt(((1)**2) + ((1)**2))
        celdas = [celda(i) for i in range(n * n)]
        voronoi = Image.new('RGB', (n, n))
        vor = voronoi.load()
        c = sns.color_palette("Set3", k).as_hex()
        for i in range(n * n):
            vor[i % n, i // n] = ImageColor.getrgb(c[celdas.pop(0)])
            limite = n
            vecinos = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx != 0 or dy !=0:
                        vecinos.append((dx, dy))
            #visual = grieta.resize((10 * n,10 * n))
            #visual.save("p4pg_{:d}.png".format(replica))
        datosActual = []
        for r in range (50):
            datosActual.append(propaga(r))
        datos.append(datosActual)
        #print(datos)

# Figuras
fig = plt.figure(figsize=(8,6))
plt.boxplot(datos60)
plt.xticks([1, 2, 3, 4, 5], ['0.2', '0.4', '0.6', '0.8'])
plt.xlabel('Número de semillas')
t = plt.title('Box plot 60')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.boxplot(datos80)
plt.xticks([1, 2, 3, 4, 5], ['0.2', '0.4', '0.6', '0.8'])
plt.xlabel('Número de semillas')
t = plt.title('Box plot 80')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.boxplot(datos100)
plt.xticks([1, 2, 3, 4, 5], ['0.2', '0.4', '0.6', '0.8'])
plt.xlabel('Número de semillas')
t = plt.title('Box plot 100')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.boxplot(datos150)
plt.xticks([1, 2, 3, 4, 5], ['0.2', '0.4', '0.6', '0.8'])
plt.xlabel('Número de semillas')
t = plt.title('Box plot 150')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.boxplot(datos150)
plt.xticks([1, 2, 3, 4, 5], ['0.2', '0.4', '0.6', '0.8'])
plt.xlabel('Número de semillas')
t = plt.title('Box plot 150 pi=0.8')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.boxplot(datos150)
plt.xticks([1, 2, 3, 4, 5], ['0.2', '0.4', '0.6', '0.8'])
plt.xlabel('Número de semillas')
t = plt.title('Box plot 150 pi=0.6')
plt.show()