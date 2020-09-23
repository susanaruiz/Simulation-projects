from random import random, randint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def caminata(currentPos, dimentions):
  nextPos = currentPos[:]
  randomDimention = randint(0, dimentions-1)
  nextPos[randomDimention] += -1 if random() < 0.5 else 1
  return nextPos

initialPos = [0] * dimentions
currentPos = initialPos

for stepNumber in range(steps):
  currentPos = caminata(currentPos, dimentions)
  if (currentPos == initialPos):
    break
  #print(currentPos)

def testing(steps, dimentions):
  initialPos = [0] * dimentions
  for t in range(steps):
    initialPos = caminata(initialPos, dimentions)
    if all([p == 0 for p in initialPos]):
      return True
  return False

steps = 2**10
dimentions = 8
total = 50
regresos = 0

for replica in range(total):
  regresos += testing(steps, dimentions)
print(stepNumber)
print(regresos, total)

#Gráfica del porciento de caminatas que no regresan y las que si
df = pd.read_csv("veces1024.csv")
df = df[['dimention', 'conteo']]
x=df['dimention']
y=df['conteo']
plt.figure(figsize=(7,7))
ax = sns.boxplot(x=x, y=y, data=df)
df.boxplot(by ='dimention', column =['conteo'], grid = False)
plt.boxplot(x, y)
plt.show
