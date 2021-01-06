import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import pandas as pd
import numpy as np

datos= pd.read_csv('results\XYCordenadas4.csv', header=None)
x=np.array(datos.iloc[:,0])
y=np.array(datos.iloc[:,1])
z=np.array(datos.iloc[:,2])
# fig = go.Figure()
# fig = go.Mesh3d(
#         x= np.array(datos.iloc[:4,0]),
#         y=np.array(datos.iloc[:4,1]),
#         z=np.array(datos.iloc[:4,2]),
#
#         i=[0, 0, 0, 1],
#         j=[1, 2, 3, 2],
#         k=[2, 3, 1, 3],
#         name='y',
#
#     )
#
# fig.add_trace =go.Mesh3d(
#         x= np.array(datos.iloc[4:8,0]),
#         y=np.array(datos.iloc[4:8,1]),
#         z=np.array(datos.iloc[4:8,2]),
#
#         i=[0, 0, 0, 1],
#         j=[1, 2, 3, 2],
#         k=[2, 3, 1, 3],
#         name='y',
#     )
# import plotly.graph_objects as go
import numpy as np

# Define random surface

fig = go.Figure()
fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[0:4,0]),
                   y=np.array(datos.iloc[0:4,1]),
                   z=np.array(datos.iloc[0:4,2]),

                  ))
fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[4:8,0]),
                   y=np.array(datos.iloc[4:8,1]),
                   z=np.array(datos.iloc[4:8,2]), i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3]))

fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[8:12,0]),
                   y=np.array(datos.iloc[8:12,1]),
                   z=np.array(datos.iloc[8:12,2]), i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3]))

fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[12:16,0]),
                   y=np.array(datos.iloc[12:16,1]),
                   z=np.array(datos.iloc[12:16,2]), i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3]))

fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[16:20,0]),
                   y=np.array(datos.iloc[16:20,1]),
                   z=np.array(datos.iloc[16:20,2]), i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3]))

fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[20:24,0]),
                   y=np.array(datos.iloc[20:24,1]),
                   z=np.array(datos.iloc[20:24,2]), i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3]))

fig.add_trace(go.Mesh3d(x=np.array(datos.iloc[24:28,0]),
                   y=np.array(datos.iloc[24:28,1]),
                   z=np.array(datos.iloc[24:28,2]), i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3]))





fig.update_layout(scene = dict(
                    xaxis_title='X Profundidad 10',
                    yaxis_title='Y Ancho 10',
                    zaxis_title='Z Alto 3.26599'),
                    width=1000,
                    margin=dict(r=20, b=10, l=10, t=10))

plot(fig)


