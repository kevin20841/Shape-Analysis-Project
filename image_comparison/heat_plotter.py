import plotly
import plotly.graph_objs as go

import pandas as pd

# Read data from a csv
z_data = pd.read_csv("../data/shape_array.csv")

data = [
    go.Surface(
        z=z_data.as_matrix()
    )
]
layout = go.Layout(
    title='3D Heatmap Representation',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='elevations-3d-surface.html')