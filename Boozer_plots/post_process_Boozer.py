#!/usr/bin/env python3
"""
This script plots the |B| contours on the plasma boundary in Boozer coordinates
"""
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from scipy.interpolate import griddata

from desc.plotting import *
from desc.compat import flip_theta


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
eq1 = flip_theta(eq1)

N = int(200)
grid = LinearGrid(L=N)
rho = np.linspace(0, 1, N + 1)

data_keys = ["iota", "D_Mercier"]

data0 = eq0.compute(data_keys, grid=grid)
data1 = eq1.compute(data_keys, grid=grid)

iota = data0["iota"]

rho0 = 1.0
fig, ax, Boozer_data0 = plot_boozer_surface(eq0, rho=rho0, return_data=True)
plt.close()

fig, ax, Boozer_data1 = plot_boozer_surface(eq1, rho=rho0, return_data=True)
plt.close()

Boozer_data_list = [Boozer_data0, Boozer_data1]

Boozer_data = Boozer_data1

theta_B0 = Boozer_data["theta_B"]
zeta_B0 = Boozer_data["zeta_B"]
B0 = Boozer_data["|B|"]

Theta = theta_B0
Zeta = zeta_B0

fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contour(
    Zeta,
    Theta,
    B0,
    levels=np.linspace(np.min(B0), np.max(B0), 30)[:],
    cmap="jet",
)

# Adding a colorbar with larger font size
cbar = fig.colorbar(contour, ax=ax, orientation="vertical")

tick_locator = ticker.MaxNLocator(nbins=6)
cbar.locator = tick_locator

cbar.ax.tick_params(labelsize=18)  # Change colorbar tick size
# cbar.set_label('|B| (T)', size=16)  # Colorbar title

# Labeling axes
ax.set_xlabel(r"$\zeta_{\mathrm{Boozer}}$", fontsize=26, labelpad=-4)
ax.set_ylabel(r"$\theta_{\mathrm{Boozer}}$", fontsize=26, labelpad=3)
ax.tick_params(axis="both", which="major", labelsize=20)  # Larger tick labels

## Adding a title and adjusting plot borders for better fit
# ax.set_title('Contour Plot of |B| in Boozer Coordinates', fontsize=18)
# plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.tight_layout()

# Increase resolution for publication quality
#plt.savefig(
#    f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_target.png",
#    dpi=300,
#)
plt.savefig(
    #f"Boozer_contour_plot_rho{rho0}_initial.pdf",
    f"Boozer_contour_plot_rho{rho0}_optimized.pdf",
    dpi=400,
)
# plt.show()
plt.close()
