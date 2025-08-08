#!/usr/bin/env python3
"""
Plot cross sections and magnetic axis position
"""
from desc import set_device

import os
import pdb
import sys
import numpy as np
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.equilibrium.coords import compute_theta_coords
import jax.numpy as jnp
from desc.grid import Grid, LinearGrid
from desc.continuation import solve_continuation_automatic
from matplotlib import pyplot as plt

from matplotlib.ticker import LogLocator, ScalarFormatter

from desc.plotting import *

comparison = True


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

files_list = [fname_path0, fname_path1]

N_equilibria = len(files_list)
color_list = ["r", "g"]
legend_list = ["initial", "optimized"]

plt.figure(figsize=(6, 5))
eq0 = Equilibrium.load(files_list[0])
eq1 = Equilibrium.load(files_list[1])

#fig, ax = plot_boundaries([eq], lw=2, color=color_list[l], legend=False)
fig, ax = plot_comparison([eq0, eq1], lw=np.array([2, 2]), phi = np.linspace(0, np.pi, 3)) #color=color_list[l], legend=False)
# Labeling axes
all_axes = fig.get_axes()
for ax in all_axes:
    ax.set_xlabel(r"$R$", fontsize=24, labelpad=0)
    ax.set_ylabel(r"$Z$", fontsize=24, labelpad=-6)
    ax.tick_params(axis="both", which="major", labelsize=18)  # Larger tick labels
    ax.title.set_fontsize(20)  # or whatever size you want


legend = fig.legends[0] if fig.legends else None
legend.set_loc('upper left') 
legend.get_frame().set_alpha(0.3)  # 0 = fully transparent, 1 = fully opaque


if legend:
    # Update the legend labels
    new_labels = ["Initial", "Optimized"]
    for text, label in zip(legend.get_texts(), new_labels):
        text.set_text(label)
        text.set_fontsize(16)

plt.savefig(f"Xsection_comparison.png", dpi=400)
#plt.show()
plt.close()
