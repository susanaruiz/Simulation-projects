from random import random, randint       #\cite SatuElisa
from math import fabs

def caminata(pos, dim):
  pos = [0] * dim           #posición de la partícula
mayor = 0
large = 10
dim = 2
for t in range(large):
  if random() < 0.5:
    pos -= 1
  else:
    pos += 1
    dist = int(fabs(pos))
    if dist > mayor:
      mayor = dist
  print(mayor)
