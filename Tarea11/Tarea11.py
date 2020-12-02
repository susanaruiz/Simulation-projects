""" Frente Pareto """

import numpy as np
import pandas as pd
from random import randint, random
import seaborn as sns
import matplotlib.pyplot as plt

def poli(maxdeg, varcount, termcount):
    f = []
    for t in range(termcount):
        var = randint(0, varcount - 1)
        deg = randint(1, maxdeg)
        f.append({'var': var, 'coef': random(), 'deg': deg})
    return pd.DataFrame(f)

def evaluate(pol, var):
    return sum([t.coef * var[pol.at[i, 'var']]**t.deg for i, t in pol.iterrows()])


def domin_by(target, challenger):
    if np.any(np.greater(target, challenger)):
        return False
    return np.any(np.greater(challenger, target))

def por(f,pob):
    return ((f * 100) / pob)
# Cambiar a 2 a 12 funciones objetivo
def objetivo(k):
    vc = 4
    md = 3
    tc = 5
    obj = [poli(md, vc, tc) for i in range(k)]
    minim = np.random.rand(k) > 0.5
    n = 250 # cuantas soluciones aleatorias
    sol = np.random.rand(n, vc)
    val = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            val[i, j] = evaluate(obj[j], sol[i])
    sign = [1 + -2 * m for m in minim]
    dom = []
    for i in range(n):
        d = [domin_by(sign * val[i], sign * val[j]) for j in range(n)]
        dom.append(not np.any(d))
    frente = val[dom, :]
    porciento = por(len(frente),n)
    return porciento
itera = 25
valores = []
for k in range(2,13,2):
    for i in range(itera):
        sim = objetivo(k)
        valores.append(sim)

df = pd.DataFrame(
    {"Función Objetivo": itera * ["02"] + itera * ["04"] + itera * ["06"] + itera * ["08"] + itera * ["10"] + itera * ["12"],
     "Porciento Pareto": valores}
     )
# Figuras
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.violinplot(x='Función Objetivo', y='Porciento Pareto', data=df, scale='count', cut = 0, palette="Set3")
sns.swarmplot(x="Función Objetivo", y="Porciento Pareto", data=df, color="white")
#sns.boxplot(x="Función Objetivo", y="Porciento Pareto", data=df, whis=np.inf, color='tan')
plt.show()


