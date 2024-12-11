import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy import sparse, interpolate

# Physical constants
hbar = 1.05457182e-34  # Planck's constant over 2π in m²·kg/s
electron_mass = 9.1093834e-31  # Electron mass in kg
effective_mass = 0.2 * electron_mass  # Effective mass in silicon

def potential_generator(
    potential_trace, x_list_to_solve, y_list_to_solve, x_list_potential, y_list_potential
):
    """
    Interpolate the potential trace onto the solution grid.

    Parameters:
    - potential_trace (2D array): Potential values on the original grid.
    - x_list_to_solve (1D array): x-coordinates of the solution grid.
    - y_list_to_solve (1D array): y-coordinates of the solution grid.
    - x_list_potential (1D array): x-coordinates of the potential grid.
    - y_list_potential (1D array): y-coordinates of the potential grid.

    Returns:
    - V (2D array): Interpolated potential on the solution grid.
    """
    # Convert potential to eV scale (if necessary)
    v_values = potential_trace / (11 * 4 * np.pi)  # Adjust conversion factor as needed

    # Create interpolator
    interp_spline = interpolate.RectBivariateSpline(
        y_list_potential, x_list_potential, v_values
    )

    # Evaluate interpolated potential on the solution grid
    X, Y = np.meshgrid(x_list_to_solve, y_list_to_solve)
    V = interp_spline.ev(Y.ravel(), X.ravel()).reshape(X.shape)

    return V

class Schrodinger2D:
    """
    Class to solve the 2D Schrödinger equation with a given potential.

    Attributes:
    - x_list_solve (1D array): x-coordinates of the solution grid.
    - y_list_solve (1D array): y-coordinates of the solution grid.
    - potential (2D array): Potential on the solution grid.
    - eigenvalues (1D array): Eigenvalues from solving the Schrödinger equation.
    - eigenstates (2D array): Corresponding eigenstates.
    """

    def __init__(self, x_list_solve, y_list_solve):
        """
        Initialize the solver with the solution grid.

        Parameters:
        - x_list_solve (1D array): x-coordinates of the solution grid.
        - y_list_solve (1D array): y-coordinates of the solution grid.
        """
        print("Assuming Dirichlet boundary conditions with boundary value zero.")
        if len(x_list_solve) != len(y_list_solve):
            print(
                "Current version supports equal number of grid points. Adjusting grids."
            )
            min_len = min(len(x_list_solve), len(y_list_solve))
            x_list_solve = np.linspace(x_list_solve[0], x_list_solve[-1], min_len)
            y_list_solve = np.linspace(y_list_solve[0], y_list_solve[-1], min_len)

        self.x_list_solve = x_list_solve
        self.y_list_solve = y_list_solve
        self.potential = None
        self.eigenvalues = None
        self.eigenstates = None

    def set_potential(self, x_list_potential, y_list_potential, potential_trace):
        """
        Set the potential on the solution grid by interpolating the given potential trace.

        Parameters:
        - x_list_potential (1D array): x-coordinates of the potential grid.
        - y_list_potential (1D array): y-coordinates of the potential grid.
        - potential_trace (2D array): Potential values on the potential grid.
        """
        # Convert potential to eV scale (if necessary)
        v_values = potential_trace / (11 * 4 * np.pi)  # Adjust conversion factor as needed

        # Create interpolator
        interp_spline = interpolate.RectBivariateSpline(
            y_list_potential, x_list_potential, v_values
        )

        # Evaluate interpolated potential on the solution grid
        X, Y = np.meshgrid(self.x_list_solve, self.y_list_solve)
        self.potential = interp_spline.ev(Y.ravel(), X.ravel()).reshape(X.shape)

    def solve_equation(self, number_of_states):
        """
        Solve the 2D Schrödinger equation for the given number of states.

        Parameters:
        - number_of_states (int): Number of eigenstates to compute.
        """
        Nx = len(self.x_list_solve)
        Ny = len(self.y_list_solve)
        dx = self.x_list_solve[1] - self.x_list_solve[0]  # in μm
        dy = self.y_list_solve[1] - self.y_list_solve[0]  # in μm
        coeff = hbar ** 2 / (2 * effective_mass) * 1e12 / 1.60218e-19  # eV·μm²

        # Construct finite difference matrices
        diag_x = np.ones(Nx)
        diags_x = np.array([diag_x, -2 * diag_x, diag_x])
        D_x = (
            -coeff
            / dx ** 2
            * sparse.spdiags(diags_x, [-1, 0, 1], Nx, Nx, format="csr")
        )

        diag_y = np.ones(Ny)
        diags_y = np.array([diag_y, -2 * diag_y, diag_y])
        D_y = (
            -coeff
            / dy ** 2
            * sparse.spdiags(diags_y, [-1, 0, 1], Ny, Ny, format="csr")
        )

        # Kinetic energy operator
        T = sparse.kronsum(D_y, D_x, format="csr")

        # Potential energy operator
        V_flat = self.potential.ravel()
        U = sparse.diags(V_flat, format="csr")

        # Hamiltonian
        H = T + U

        # Solve eigenvalue problem
        self.eigenvalues, self.eigenstates = eigsh(
            H, k=number_of_states, which="SM"
        )

    def plot_state(self, idx_to_plot):
        """
        Plot the probability density of the specified eigenstate overlaid with the potential.

        Parameters:
        - idx_to_plot (int): Index of the eigenstate to plot.
        """
        Nx = len(self.x_list_solve)
        Ny = len(self.y_list_solve)
        X, Y = np.meshgrid(self.x_list_solve, self.y_list_solve)

        # Reshape eigenstate
        psi = self.eigenstates[:, idx_to_plot].reshape(Ny, Nx)
        probability_density = np.abs(psi) ** 2

        # Plot probability density
        fig, ax = plt.subplots()
        c = ax.pcolor(X, Y, probability_density, shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax, label="Probability Density")

        # Overlay potential
        cs = ax.contour(
            X, Y, self.potential, colors="white", linewidths=0.5, alpha=0.7
        )
        ax.set_title(f"Eigenstate {idx_to_plot}")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        plt.show()
