from random import random, randint            #\cite SatuElisa
from math import fabs

def caminata(pos, dim):
  x = randint(0, dim - 1)
  pos[x] += -1 if random() < 0.5 else 1
  return pos

def testing(large, dim):
  pos = [0] * dim
  for t in range(large):
    pos = caminata(pos, dim)
    if all([p == 0 for p in pos]):
      return True
  return False   

large = 1024
dim = 1
total = 50
regresos = 0
for replicas in range(total):
  regresos += testing(large, dim)
print(regresos / total)

