#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
from desc.utils import setdefault
from desc.vmec import VMECIO


class NeoIO:
    """Class to interface with NEO."""

    def __init__(self, name, eq, ns=256, M_booz=None, N_booz=None):
        self.name = name
        self.vmec_file = f"wout_{name}.nc"
        self.booz_file = f"boozmn.{name}"
        self.neo_in_file = f"neo_in.{name}"
        self.neo_out_file = f"neo_out.{name}"

        self.eq = eq
        self.ns = ns  # number of surfaces
        self.M_booz = setdefault(M_booz, 3 * eq.M + 1)
        self.N_booz = setdefault(N_booz, 3 * eq.N)

    @staticmethod
    def read(name):
        """Return ρ and ε¹ᐧ⁵ from NEO output with given name."""
        neo_eps = np.loadtxt(name)[:, 1]
        neo_rho = np.sqrt(np.linspace(1 / (neo_eps.size + 1), 1, neo_eps.size))
        # replace bad values with linear interpolation
        good = np.isfinite(neo_eps)
        neo_eps[~good] = np.interp(neo_rho[~good], neo_rho[good], neo_eps[good])
        return neo_rho, neo_eps

    def write(self):
        """Write neo input file."""
        print(f"Writing VMEC wout to {self.vmec_file}")
        VMECIO.save(self.eq, self.vmec_file, surfs=self.ns, verbose=0)
        self._write_booz()
        self._write_neo()

    def _write_booz(self):
        print(f"Writing boozer output file to {self.booz_file}")
        import booz_xform as bx

        b = bx.Booz_xform()
        b.read_wout(self.vmec_file)
        b.mboz = self.M_booz
        b.nboz = self.N_booz
        b.run()
        b.write_boozmn(self.booz_file)

    def _write_neo(
        self,
        theta_n=200,
        phi_n=200,
        num_pitch=50,
        multra=1,
        acc_req=0.01,
        nbins=100,
        nstep_per=75,
        nstep_min=500,
        nstep_max=2000,
        verbose=2,
    ):
        print(f"Writing NEO input file to {self.neo_in_file}")
        f = open(self.neo_in_file, "w")

        def writeln(s):
            f.write(str(s))
            f.write("\n")

        # https://princetonuniversity.github.io/STELLOPT/NEO
        writeln(f"'#' {datetime.now()}")
        writeln(f"'#' {self.vmec_file}")
        writeln(f"'#' M_booz={self.M_booz}. N_booz={self.N_booz}.")
        writeln(self.booz_file)
        writeln(self.neo_out_file)
        # Neo computes things on the so-called "half grid" between the full grid.
        # There are only ns - 1 surfaces there.
        writeln(self.ns - 1)
        # NEO starts indexing at 1 and does not compute on axis (index 1).
        surface_indices = " ".join(str(i) for i in range(2, self.ns + 1))
        writeln(surface_indices)
        writeln(theta_n)
        writeln(phi_n)
        writeln(0)
        writeln(0)
        writeln(num_pitch)
        writeln(multra)
        writeln(acc_req)
        writeln(nbins)
        writeln(nstep_per)
        writeln(nstep_min)
        writeln(nstep_max)
        writeln(0)
        writeln(verbose)
        writeln(0)
        writeln(0)
        writeln(2)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln("'#'\n'#'\n'#'")
        writeln(0)
        writeln(f"neo_cur.{self.name}")
        writeln(200)
        writeln(2)
        writeln(0)
        f.close()




eq = get("W7-X")
rho = np.linspace(0, 1, 50)
#grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
grid = LinearGrid(rho=rho, theta=eq.M_grid, zeta=eq.N_grid, NFP=eq.NFP, sym=False)
num_transit = 10
data = eq.compute(
    "effective ripple 3/2",
    grid=grid,
    theta=Bounce2D.compute_theta(eq, X=32, Y=64, rho=rho),
    Y_B=128,
    num_transit=num_transit,
    num_well=20 * num_transit,
)

#assert np.isfinite(data["effective ripple 3/2"]).all()
eps_32 = grid.compress(data["effective ripple 3/2"])
neo_rho, neo_eps_32 = NeoIO.read("/home/rgaur/DESC/tests/inputs/neo_out.w7x")
#np.testing.assert_allclose(eps_32, np.interp(rho, neo_rho, neo_eps_32), rtol=0.16)


# Create the plot
plt.figure(figsize=(8, 6))

# Plot both curves
plt.plot(rho, eps_32, '-', linewidth=3, color='r', label="DESC")
plt.plot(neo_rho, neo_eps_32, '-', linewidth=3, color='b', label='NEO')

# Set labels and title
plt.xlabel(r'$\rho$', fontsize=24)
plt.ylabel(r'$\epsilon_{eff}^{3/2}$', fontsize=24)
#plt.title('W7-X effective ripple')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# Set axis limits
plt.xlim(0.0, 1.0)
plt.ylim(0.0003, 0.001)

# Add legend
plt.legend(fontsize=20)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig('w7x_ripple.png', dpi=300, bbox_inches='tight')
plt.show()


