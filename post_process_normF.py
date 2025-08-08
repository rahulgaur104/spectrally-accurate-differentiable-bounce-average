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

# keyword_arr = ["OT", "OH", "OP"]
keyword_arr = ["OP"]

fname_path0 = (
    os.path.dirname(os.getcwd())
    + "/eq_final_OH.h5"
)
fname_path1 = (
    os.path.dirname(os.getcwd())
    + "/opt_step_11.h5"
)
eq0 = Equilibrium.load(f"{fname_path0}")
eq1 = Equilibrium.load(f"{fname_path1}")

files_list = [fname_path0, fname_path1]

N_equilibria = len(files_list)
color_list = ["r", "g"]
legend_list = ["initial", "optimized"]

plt.figure(figsize=(6, 5))
eq = Equilibrium.load(files_list[l])
fig, ax = plot_section(eq, name="|F|", norm_F=True, log=True)
# Labeling axes
plt.tight_layout()
plt.savefig(f"normF/{keyword}_normF_plot_{legend_list[l]}.png", dpi=400)
plt.close()

plt.figure(figsize=(6, 5))
eq = Equilibrium.load(files_list[l])
fig, ax = plot_section(eq, name="|F|", norm_F=True)
# Labeling axes
plt.tight_layout()
plt.savefig(f"normF/{keyword}_normF_plot_nolog_{legend_list[l]}.png", dpi=400)
plt.close()
