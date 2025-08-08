#!/usr/bin/env python3

import os
import pdb
import numpy as np
import pickle
import matplotlib.pyplot as plt

from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals import Bounce2D

from desc.equilibrium import Equilibrium

fname_path0 = (
    os.path.dirname(os.getcwd())
    + "/eq_initial.h5"
)
fname_path1 = (
    os.path.dirname(os.getcwd())
    + "/eq_optimized_final2.h5"
)
rho = np.linspace(1e-12, 1, 20)
eq0 = Equilibrium.load(f"{fname_path0}")
eq1 = Equilibrium.load(f"{fname_path1}")

eq_list = [eq0, eq1]
grid = LinearGrid(rho=rho, theta=eq0.M_grid, zeta=eq0.N_grid, NFP=eq0.NFP, sym=False)

data_arr = np.zeros((2, len(rho)))
# These are higher resolution than needed.
num_transit = 16

res = np.array([32, 64, 24, 50])

for i, eq in enumerate(eq_list):
    data = eq.compute(
        "effective ripple 3/2",
        grid=grid,
        theta=Bounce2D.compute_theta(eq, X=res[0], Y=res[1], rho=rho),
        num_transit=num_transit,
        num_quad=res[2],
        num_pitch=res[3],
        num_well=20 * num_transit,
    )
    
    eps_32 = grid.compress(data["effective ripple 3/2"])
    data_arr[i, :] = eps_32

fig, ax = plt.subplots(figsize=(6, 5))

plt.plot(rho, data_arr[0, :], '-or', linewidth=3,  ms=2, label="initial")
plt.plot(rho, data_arr[1, :], '-ob', linewidth=3,  ms=2, label="optimized")

# Set labels and title
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\epsilon_{\mathrm{eff}}^{3/2}$', fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Set axis limits
plt.xlim(0.0, 1.0)
#plt.ylim(0.0003, 0.001)

# Add legend
plt.legend(fontsize=20)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout
plt.tight_layout()

# Save the plot
#plt.savefig('w7x_ripple.pdf', dpi=300, bbox_inches='tight')
plt.savefig('ripple_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()




