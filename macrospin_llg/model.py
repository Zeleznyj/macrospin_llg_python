import numpy as np
import scipy.integrate
import scipy.optimize

from multiprocess import Pool as _MultiprocessPool

from .solution import Solution
from .minimizers import MinimizerMixin



class Model(MinimizerMixin):
    """
    Python model for solving the Landau-Lifshitz-Gilbert (LLG) equations.
    This class represents the magnetic moments as a 2D array of shape (n_mag, 3).
    """
    # Physical Constants
    mu_B = 5.7883818012e-5  # Bohr Magneton (eV/T)
    gamma_e = 1.760859644e11  # Gyromagnetic ratio (rad/s*T)

    def __init__(self, n_mag):
        """
        Initializes the model for a given number of magnetic moments.

        Args:
            n_mag (int): The number of magnetic moments in the system.
        """
        if n_mag <= 0:
            raise ValueError("n_mag must be a positive integer")
        self.n_mag = int(n_mag)

        # Hamiltonian and model parameters
        self.J = np.zeros((3, 3, self.n_mag, self.n_mag))
        self.K2 = np.zeros(self.n_mag)
        self.A2 = np.zeros((3, self.n_mag))
        self.K4 = np.zeros(self.n_mag)
        self.K6 = np.zeros(self.n_mag)
        self.A6 = np.zeros((3, self.n_mag))
        self.D = np.zeros((3, 3, self.n_mag, self.n_mag))
        self.ag = 0.0  # Gilbert damping parameter

        # Effective field terms are shaped (n_mag, 3)
        self.B = np.zeros((self.n_mag, 3))
        # New: multiple custom interactions
        # Per-spin dict of Beff callables by name: name -> f(t, M) -> (3,)
        self._custom_beff_terms = [dict() for _ in range(self.n_mag)]
        # Global dict of custom energy callables by name: name -> g(t, M) -> float
        self._custom_energy_terms = {}

    def _ita(self, a):
        """
        Internal helper to convert user-facing indices to zero-based list. -1 means all moments.
        Only valid indices (0 <= a < self.n_mag, or list thereof, or -1) are accepted.
        Raises ValueError otherwise.
        """
        if np.isscalar(a):
            a = int(a)
            if a == -1:
                return range(self.n_mag)
            if 0 <= a < self.n_mag:
                return [a]
            else:
                raise ValueError(f"Invalid moment index {a}: must be in 0..{self.n_mag-1}, or -1 for all.")
        else:
            a_list = [int(x) for x in a]
            if len(a_list) == 0:
                raise ValueError("Empty index list is not allowed.")
            for x in a_list:
                if not (0 <= x < self.n_mag):
                    raise ValueError(f"Invalid moment index {x}: must be in 0..{self.n_mag-1}, or -1 for all.")
            return a_list

    def add_exchange(self, a, b, J):
        """Adds exchange interaction between moments a and b. Uses zero-based indexing. -1 means all moments."""
        if np.isscalar(J):
            Jm = np.diag([J, J, J])
        else:
            Jm = np.asarray(J)

        it1, it2 = self._ita(a), self._ita(b)
        for i in it1:
            for j in it2:
                if i != j:
                    self.J[:, :, i, j] = Jm
                    self.J[:, :, j, i] = Jm  # Ensure symmetry

    def add_ani2(self, a, K, A):
        """Adds uniaxial anisotropy (K*(M.A)^2). Zero-based indexing, -1 for all moments."""
        it = self._ita(a)
        A_norm = np.asarray(A) / np.linalg.norm(A)
        for i in it:
            self.K2[i] = K
            self.A2[:, i] = A_norm

    def add_ani4(self, a, K):
        """Adds cubic anisotropy (K*(Mx^4+My^4+Mz^4)). Zero-based indexing, -1 for all moments."""
        it = self._ita(a)
        for i in it:
            self.K4[i] = K

    def add_ani6(self, a, K):
        """Adds sixth-order anisotropy (K*(M.A)^6). Zero-based indexing, -1 for all moments."""
        it = self._ita(a)
        for i in it:
            self.K6[i] = K

    def add_B(self, a, B):
        """Adds a static magnetic field B (in Tesla). Zero-based indexing, -1 for all moments."""
        it = self._ita(a)
        for i in it:
            self.B[i, :] = B

    def add_DMI(self, a, b, d):
        """Adds Dzyaloshinskii-Moriya interaction between a and b (zero-based indices)."""
        a_idx, b_idx = int(a), int(b)  # both a and b are now zero-based indices
        d_vec = np.asarray(d)
        d_matrix = np.array([[0, -d_vec[2], d_vec[1]],
                             [d_vec[2], 0, -d_vec[0]],
                             [-d_vec[1], d_vec[0], 0]])
        self.D[:, :, a_idx, b_idx] += d_matrix
        self.D[:, :, b_idx, a_idx] -= d_matrix  # DMI is anti-symmetric

    def add_custom_interaction(self, name, beff_per_atom=None, energy=None):
        """Adds a custom interaction.

        Args:
            name (str): Unique name for this custom interaction.
            beff_per_atom (sequence|None): Sequence of length n_mag where each element is a callable
                beff_i(t, M) -> (3,) for spin i (or None to skip that spin).
            energy (callable|None): Function energy(t, M) -> float contributing to total energy.

        Provide at least one of beff_per_atom or energy. Supplying both corresponds to a conservative
        interaction where beff is consistent with the gradient of the provided energy.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string.")
        if beff_per_atom is None and energy is None:
            raise ValueError("At least one of beff_per_atom or energy must be provided.")

        if beff_per_atom is not None:
            try:
                beff_list = list(beff_per_atom)
            except TypeError:
                raise TypeError("beff_per_atom must be a sequence of length n_mag with callables or None.")
            if len(beff_list) != self.n_mag:
                raise ValueError(f"beff_per_atom must have length n_mag={self.n_mag}.")
            for i, beff_i in enumerate(beff_list):
                if beff_i is None:
                    continue
                if not callable(beff_i):
                    raise TypeError(f"beff_per_atom[{i}] must be callable or None.")
                self._custom_beff_terms[i][name] = beff_i

        if energy is not None:
            # Energy terms are global contributions; add once
            self._custom_energy_terms[name] = energy

    # Note: legacy add_Bmf has been removed. Use add_custom_interaction instead.

    def B_ani(self, n, M):
        """Calculates the anisotropy field for the n-th moment."""
        Mn = M[n, :]
        B_a = np.zeros(3)
        norm_Mn_sq = np.dot(Mn, Mn)
        B_a += self.K2[n] * self.A2[:, n] * np.dot(Mn, self.A2[:, n]) / norm_Mn_sq
        B_a += 2 * self.K4[n] * Mn ** 3 / (norm_Mn_sq ** 2)
        B_a += self.K6[n] / (2*(norm_Mn_sq ** 3)) * np.array([
            6*Mn[0]**5 - 15 * 4 * Mn[0]**3 * Mn[1]**2 + 15 * 2 * Mn[0]*Mn[1]**4,
            -6*Mn[1]**5 - 15 * 2 * Mn[0]**4 * Mn[1] + 15 * 4 * Mn[0]**2 * Mn[1]**3,
            0
        ])
        return B_a / self.mu_B

    def B_ex(self, n, M):
        """Calculates the exchange field for the n-th moment."""
        B_J = np.zeros(3)
        Mn = M[n, :]
        norm_Mn = np.linalg.norm(Mn)
        for m in range(self.n_mag):
            if n != m:
                Mm = M[m, :]
                norm_Mm = np.linalg.norm(Mm)
                B_J -= 2 * self.J[:, :, n, m] @ Mm / (norm_Mn * norm_Mm * self.mu_B)
        return B_J

    def B_DMI(self, n, M):
        """Calculates the DMI field for the n-th moment."""
        B_D = np.zeros(3)
        Mn = M[n, :]
        norm_Mn = np.linalg.norm(Mn)
        for m in range(self.n_mag):
            if n != m:
                Mm = M[m, :]
                norm_Mm = np.linalg.norm(Mm)
                B_D -= self.D[:, :, n, m] @ Mm / (norm_Mn * norm_Mm * self.mu_B)
        return B_D

    def Beff(self, t, n, M):

        B = np.zeros(3)
        B += self.B_ani(n, M)
        B += self.B_ex(n, M)
        B += self.B_DMI(n,M)
        B += self.B[n, :]
        # Sum all custom Beff terms
        if self._custom_beff_terms[n]:
            for term in self._custom_beff_terms[n].values():
                B += term(t, M)

        return B



    def find_dM0(self, t0, M0):
        """
        Not used anywhere, probably wrong !!!!
        """

        gamma_prime = self.gamma_e * 1e-9

        dM0 = np.zeros((self.n_mag, 3))
        for n in range(self.n_mag):
            Beff = self.Beff(t0, n, M0)
            Mn = M0[n, :]
            norm_Mn = np.linalg.norm(Mn)
            dM0n = -gamma_prime * np.cross(Mn, Beff) + self.ag * gamma_prime * Beff * norm_Mn
            dM0n = dM0n / (1 + self.ag**2 )
            dM0[n, :] = dM0n

        return dM0

    def LLG_implicit(self, t, M_flat, dM_flat):
        """
        The implicit LLG residual function, F(t, y, y') = 0.
        This is the internal interface to the solver, which works with flat 1D arrays.
        """
        M = M_flat.reshape((self.n_mag, 3))
        dM = dM_flat.reshape((self.n_mag, 3))

        f = np.zeros_like(M)
        gamma_prime = self.gamma_e * 1e-9  # Use GHz/T for better scaling

        for n in range(self.n_mag):
            Mn = M[n, :]
            dMn = dM[n, :]
            norm_Mn = np.linalg.norm(Mn)

            res = dMn / gamma_prime
            res -= self.ag * np.cross(Mn / norm_Mn, dMn / gamma_prime)

            Beff = self.Beff(t, n, M)
            res += np.cross(Mn, Beff) 

            f[n, :] = res

        return f.flatten()

    def LLG_explicit(self, t, M_flat):

        M = M_flat.reshape((self.n_mag, 3))

        f = np.zeros_like(M)
        gamma_prime = self.gamma_e / (1 + self.ag**2)

        for n in range(self.n_mag):
            Mn = M[n, :]
            norm_Mn = np.linalg.norm(Mn)

            Beff = self.Beff(t, n, M)

            f[n, :] -= gamma_prime * np.cross(Mn, Beff)

            f[n, :] -= gamma_prime * self.ag / norm_Mn * np.cross(Mn, np.cross(Mn, Beff))

        return f.flatten() * 1e-9

    def solve_LLG(self, tf, M0, t0=0.0, method='DOP853', t_eval=None, **solver_kwargs):
        """
        Solves the LLG equations using an explicit solver via scipy.integrate.solve_ivp.

        Args:
            tf (float): End time for the simulation (in seconds).
            M0 (np.ndarray): Initial magnetic configuration, shape (n_mag, 3).
            t0 (float, optional): Start time. Defaults to 0.0.
            method (str, optional): Integration method to use. Defaults to 'DOP853'.
            t_eval (np.ndarray, optional): Times at which to store the computed solution.
            **solver_kwargs: Additional keyword arguments passed to solve_ivp.

        Returns:
            A solution object with attributes:
            .t (np.ndarray): 1D array of time points.
            .y (np.ndarray): 3D array of states, shape (time_points, n_mag, 3).
        """


        M0 = np.asarray(M0)
        if M0.shape != (self.n_mag, 3):
            raise ValueError(f"M0 shape must be ({self.n_mag}, 3), but got {M0.shape}")

        # Wrapper for solve_ivp
        def fun(t, y):
            return self.LLG_explicit(t, y)

        # If t_eval is not provided, we might want to generate some points or let solve_ivp decide.
        # However, Solution expects a reasonable number of points.
        # If t_eval is None, solve_ivp returns points chosen by the solver.
        
        res = scipy.integrate.solve_ivp(
            fun,
            (t0, tf),
            M0.flatten(),
            method=method,
            t_eval=t_eval,
            **solver_kwargs
        )

        if not res.success:
            print(f"Warning: Solver failed: {res.message}")

        # Reshape output
        # res.y has shape (n_vars, n_times) -> need (n_times, n_mag, 3)
        M_reshaped = res.y.T.reshape((len(res.t), self.n_mag, 3))

        return Solution(self, res.t, M_reshaped)

    def solve_LLG_implicit(self, tf, M0, t0=0.0, ncp=100, solver_kwargs=None):
        """
        Solves the LLG equations using an implicit DAE solver.

        Args:
            tf (float): End time for the simulation (in seconds).
            M0 (np.ndarray): Initial magnetic configuration, shape (n_mag, 3).
            t0 (float, optional): Start time. Defaults to 0.0.
            ncp (int, optional): Number of communication points (output steps).
            solver_kwargs (dict, optional): Dictionary of options directly applied
                as attributes on the underlying solver instance (e.g. 'rtol', 'atol',
                'maxsteps', etc.). No validation is performed.

        Returns:
            A solution object with attributes:
            .t (np.ndarray): 1D array of time points.
            .y (np.ndarray): 3D array of states, shape (time_points, n_mag, 3).
        """
        M0 = np.asarray(M0)
        if M0.shape != (self.n_mag, 3):
            raise ValueError(f"M0 shape must be ({self.n_mag}, 3), but got {M0.shape}")

        #dM0 = self.find_dM0(t0, M0)
        dM0 = self.LLG_explicit(t0, M0)

        try:
            import assimulo.problem as apr
            import assimulo.solvers as aso
        except ImportError as exc:
            raise ImportError(
                "The 'assimulo' package is required for the implicit solver (solve_LLG_implicit). "
                "Please install it via 'pip install macrospin_llg[assimulo]' or 'pip install assimulo'."
            ) from exc

        def res(t, y, yd):
            return self.LLG_implicit(t, y, yd)

        prob = apr.Implicit_Problem(res, M0.flatten(), dM0.flatten(), t0)
        sim = aso.IDA(prob)

        # Directly apply provided kwargs to solver instance
        if isinstance(solver_kwargs, dict):
            for key, value in solver_kwargs.items():
                setattr(sim, key, value)

        t, y_flat, dy_flat = sim.simulate(tf, ncp=ncp)

        # Reshape the flat output from the solver into the 3D array
        num_timesteps = len(t)
        M_reshaped = y_flat.reshape((num_timesteps, self.n_mag, 3))

        return Solution(self, np.array(t), np.array(M_reshaped))

    def energy(self, t, M):
        """
        Calculates the total energy and its components for a given magnetic state.

        Args:
            t (float): The time (for time-dependent fields).
            M (np.ndarray): The magnetic configuration, shape (n_mag, 3).

        Returns:
            dict: A dictionary of energy contributions with keys:
                'total', 'exchange', 'dmi', 'anisotropy', 'b_field', plus one key per custom term name.
        """
        E_exchange = float(self._energy_exchange(M))
        E_dmi = float(self._energy_dmi(M))
        E_anisotropy = float(self._energy_anisotropy(M))
        E_b_field = float(self._energy_b_field(t, M))
        contributions = {
            'exchange': E_exchange,
            'dmi': E_dmi,
            'anisotropy': E_anisotropy,
            'b_field': E_b_field,
        }
        for name, g in self._custom_energy_terms.items():
            contributions[name] = float(g(t, M))
        E_total = sum(contributions.values())
        contributions['total'] = E_total
        return contributions

    def _energy_exchange(self, M):
        E = 0.0
        for i in range(self.n_mag):
            for m in range(self.n_mag):  # Loop over unique pairs
                if i != m:
                    Mi, Mm = M[i, :], M[m, :]
                    norm_Mi = np.linalg.norm(Mi)
                    norm_Mm = np.linalg.norm(Mm)
                    E += Mi @ self.J[:, :, i, m] @ Mm / (norm_Mi * norm_Mm)
        return E

    def _energy_dmi(self, M):
        E = 0.0
        for i in range(self.n_mag):
            for m in range(self.n_mag):  # Loop over unique pairs
                if i != m:
                    Mi, Mm = M[i, :], M[m, :]
                    norm_Mi = np.linalg.norm(Mi)
                    norm_Mm = np.linalg.norm(Mm)
                    E += Mi @ self.D[:, :, i, m] @ Mm / (norm_Mi * norm_Mm)
        return E

    def _energy_anisotropy(self, M):
        E = 0.0
        for i in range(self.n_mag):
            Mi = M[i, :]
            norm_Mi_sq = np.dot(Mi, Mi)
            # Uniaxial
            E -= self.K2[i] / 2 * (np.dot(Mi, self.A2[:, i])) ** 2 / norm_Mi_sq
            # Cubic
            E -= self.K4[i] / 2 * np.sum(Mi ** 4) / (norm_Mi_sq ** 2)
            # Sixth-order
            E -= self.K6[i] / 2 * (Mi[0]**6 - Mi[1]**6 - 15*Mi[0]**4*Mi[1]**2 + 15*Mi[0]**2*Mi[1]**4)/ (norm_Mi_sq ** 3)
        return E

    def _energy_b_field(self, t, M):
        # Sums the dot product of each moment with its corresponding field
        E = -np.sum(M * self.B) * self.mu_B
        return E
