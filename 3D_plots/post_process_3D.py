#!/usr/bin/env python3

import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from scipy.interpolate import griddata

from desc.plotting import *

# from plotly.io import write_html


fname_path0 = (
    os.path.dirname(os.getcwd())
    + "/eq_initial.h5"
)
fname_path1 = (
    os.path.dirname(os.getcwd())
    + "/eq_optimized_final2.h5"
)

eq0 = Equilibrium.load(f"{fname_path0}")
eq1 = Equilibrium.load(f"{fname_path1}")

legend_list = ["initial", "optimized"]
eq_list = [eq0, eq1]
scale_list = [2, 4]

for eq, legend, scale in zip(eq_list, legend_list, scale_list):
    plt.figure()
    theta_grid = np.linspace(0, 2 * np.pi, 300)
    zeta_grid = np.linspace(0, 2 * np.pi, 300)
    grid = LinearGrid(rho=1.0, theta=theta_grid, zeta=zeta_grid)
    fig = plot_3d(eq, name="|B|", grid=grid)

    # fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='lightgray', linewidth=2, linecolor='black')
    # fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='lightgray', linewidth=2, linecolor='black')
    # fig.update_layout(coloraxis_colorbar=dict(len=0.8, thickness=20))
    fig.update_traces(
        colorbar=dict(
            tickfont=dict(size=58),  # Adjust the size value as needed
            title_font=dict(size=58),  # Adjust the size value as needed
        )
    )

    fig.update_layout(font=dict(size=18, color="black", family="Arial, sans-serif"))

    config = {
        "toImageButtonOptions": {
            "filename": f"modB_3d_{legend}",
            "format": "svg",
            "scale": scale,
        }
    }

    save_path_html = os.getcwd() + f"/modB_3d_{legend}.html"
    fig.write_html(
        save_path_html, config=config, include_plotlyjs=True, full_html=True
    )
    
    # save_path_png = os.getcwd() + f"/3D_modB/modB_3d_{keyword}_{legend}.png"
    # fig.write_image(save_path_png, scale=scale)
    plt.close()
