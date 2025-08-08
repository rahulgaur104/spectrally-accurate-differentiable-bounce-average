import numpy as np
import pickle
import matplotlib.pyplot as plt

from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
#from tests.test_neoclassical import NeoIO


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


# Create the plot
plt.figure(figsize=(8, 6))


eq = get("W7-X")
rho = np.linspace(1e-12, 1, 50)

grid = LinearGrid(rho=rho, theta=eq.M_grid, zeta=eq.N_grid, NFP=eq.NFP, sym=False)

# These are higher resolution than needed.
num_transit = 32

res_table = [np.array([16, 32, 12, 25]), np.array([32, 64, 24, 50]), np.array([32, 64, 48, 100]), np.array([64, 128, 48, 100])]
#res_table = [np.array([16, 32, 12, 25]), np.array([32, 64, 24, 50])]
markers = ['-r', '-g', '-b', '-k']
#legend_list = [""]


for i, res in enumerate(res_table):

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
    #data = {"eps_32": eps_32, "rho": rho}
    np.save(f"eps_32_{i}.npy", eps_32)

    plt.plot(rho, eps_32, markers[i], linewidth=3, label=f"X={res[0]},Y_B={res[1]}," + r"$N_{b^{'}}$" + f"={res[3]}" + r",$N_{\mathrm{quad}}$" + f"={res[2]}," + r"$N_{l}$=20")



#with open(name, "wb") as f:
#    pickle.dump(data, f)



## Plot both curves
#plt.plot(rho, eps_32, '-s', linewidth=3, color='r', label="DESC")

# Set labels and title
plt.xlabel(r'$\rho$', fontsize=26)
plt.ylabel(r'$\epsilon_{\mathrm{eff}}^{3/2}$', fontsize=26)
#plt.title('W7-X effective ripple')

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Set axis limits
plt.xlim(0.0, 1.0)
plt.ylim(0.0003, 0.001)

# Add legend
plt.legend(fontsize=24)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout
plt.tight_layout()

# Save the plot
#plt.savefig('w7x_ripple.pdf', dpi=300, bbox_inches='tight')
plt.savefig('w7x_ripple_resolution.pdf', dpi=300, bbox_inches='tight')
plt.show()




#def load_and_plot(name):
#    with open(name, "rb") as f:
#        data = pickle.load(f)
#
#    eps_32 = data["eps_32"]
#    rho = data["rho"]
#    neo_rho, neo_eps_32 = NeoIO.read("/home/rgaur/DESC/tests/inputs/neo_out.w7x")
#
#    #fig, ax = plt.subplots()
#    #ax.plot(rho, eps_32, marker="o")
#    #ax.plot(neo_rho, neo_eps_32)
#    #plt.show()
#
#    # Create the plot
#    plt.figure(figsize=(8, 6))
#    
#    # Plot both curves
#    plt.plot(rho, eps_32, '-s', linewidth=3, color='r', label="DESC")
#    plt.plot(neo_rho, neo_eps_32, '-o', linewidth=3, color='b', label='NEO')
#    
#    # Set labels and title
#    plt.xlabel(r'$\rho$', fontsize=26)
#    plt.ylabel(r'$\epsilon_{\mathrm{eff}}^{3/2}$', fontsize=26)
#    #plt.title('W7-X effective ripple')
#    
#    plt.xticks(fontsize=24)
#    plt.yticks(fontsize=24)
#    
#    # Set axis limits
#    plt.xlim(0.0, 1.0)
#    plt.ylim(0.0003, 0.001)
#    
#    # Add legend
#    plt.legend(fontsize=24)
#    
#    # Add grid
#    plt.grid(True, linestyle='--', alpha=0.7)
#    
#    # Tight layout
#    plt.tight_layout()
#    
#    # Save the plot
#    plt.savefig('w7x_ripple.pdf', dpi=300, bbox_inches='tight')
#    plt.show()
#
#
#
##name = "nov26w7x.pkl"
#name = "desc_eps_32.pkl"
## compute_and_save(name)
#load_and_plot(name)
