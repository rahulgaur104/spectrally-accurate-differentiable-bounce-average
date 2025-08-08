#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt


x = np.linspace(0, 1, 100)

x0 = 10

plt.plot(x, x, label=f"$x_l$ = {x0}," + r"$y_{\mathrm{b}}$" + "= 20")

# Add legend
plt.legend(fontsize=20)

plt.show()
