from google.colab import files
files.upload()

import pandas as pd
df = pd.read_csv("datos.csv")
print(df.head())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
  
df = pd.read_csv("datos.csv")
df = df[['1', '0.88']]

x=df['1']
y=df['0.88']
plt.figure(figsize=(7,7))
ax = sns.boxplot(x=x, y=y, data=df)
#df.boxplot(by ='1', column =['0.88'], grid = False)
#plt.boxplot(x, y)
plt.show()