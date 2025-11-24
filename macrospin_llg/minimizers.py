import numpy as np
import scipy.optimize
from multiprocess import Pool as _MultiprocessPool
import time
from tqdm import tqdm

def _angles_to_M(angles, target_norms, n_mag):
    M = np.zeros((n_mag, 3), dtype=float)
    for i in range(n_mag):
        theta = angles[2*i + 0]
        phi = angles[2*i + 1]
        r = float(target_norms[i])
        st = np.sin(theta)
        M[i, 0] = r * st * np.cos(phi)
        M[i, 1] = r * st * np.sin(phi)
        M[i, 2] = r * np.cos(theta)
    return M

def _M_to_angles(M, target_norms, n_mag):
    ang = np.zeros(2*n_mag, dtype=float)
    for i in range(n_mag):
        x, y, z = M[i, 0], M[i, 1], M[i, 2]
        r = float(target_norms[i])
        z_clamped = np.clip(z / r, -1.0, 1.0)
        theta = np.arccos(z_clamped)
        phi = np.arctan2(y, x)
        ang[2*i + 0] = theta
        ang[2*i + 1] = phi
    return ang


def _run_parallel_task(model, mode, t, start, solver_kwargs, llg_kwargs):
    """
    Executes either the angle-based minimization, Cartesian minimization, or LLG dynamics based on mode.

    Returns:
        tuple: (M_opt, total_energy, aux_result)
    """
    mode = str(mode).lower()
    if mode == 'minimize_angles':
        M_opt, energy_dict, opt_res = model.minimize_energy_angles(
            t, start, solver_kwargs=solver_kwargs
        )
        total_energy = float(energy_dict['total'])
        return M_opt, total_energy, opt_res
    elif mode == 'minimize':
        M_opt, energy_dict, opt_res = model.minimize_energy(
            t, start, solver_kwargs=solver_kwargs
        )
        total_energy = float(energy_dict['total'])
        return M_opt, total_energy, opt_res
    elif mode == 'llg':
        llg_opts = dict(llg_kwargs) if llg_kwargs else {}
        if 'tf' not in llg_opts:
            raise ValueError("llg_kwargs must include 'tf' when mode='llg'.")
        tf = llg_opts.pop('tf')
        t0 = llg_opts.pop('t0', t)
        solution = model.solve_LLG(tf=tf, M0=start, t0=t0, **llg_opts)
        M_final = solution.M[-1]
        energy_time = solution.t[-1]
        total_energy = float(model.energy(energy_time, M_final)['total'])
        return M_final, total_energy, solution
    else:
        raise ValueError("mode must be 'minimize', 'minimize_angles', or 'llg'.")


def _parallel_task_worker(payload):
    """
    Helper for multiprocess pools.
    payload = (model, mode, t, start, solver_kwargs, llg_kwargs)
    """
    model, mode, t, start, solver_kwargs, llg_kwargs, idx = payload
    res = _run_parallel_task(model, mode, t, start, solver_kwargs, llg_kwargs)
    return (idx,) + res

class MinimizerMixin:
    """
    Mixin class providing energy minimization capabilities for the Model.
    Expects 'self' to have:
      - n_mag (int)
      - mu_B (float)
      - energy(t, M) -> dict
      - Beff(t, n, M) -> np.ndarray
      - solve_LLG(...) [if parallel_minimize_energy mode='llg' is used]
    """

    def gradient_angles(self, t, ang, target_norms):
        M = _angles_to_M(ang, target_norms, self.n_mag)
        grad = np.zeros(2*self.n_mag)
        mu_B = self.mu_B
        for n in range(self.n_mag):
            theta = ang[2*n]
            phi = ang[2*n + 1]
            angles_der = np.array([
                np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                -np.sin(theta)
            ])
            angles_der2 = np.array([
                -np.sin(theta) * np.sin(phi),
                np.sin(theta) * np.cos(phi),
                0
            ])
            Beff_n = self.Beff(t, n, M)
            grad[2*n] = -mu_B * target_norms[n] * np.dot(Beff_n, angles_der)
            grad[2*n + 1] = -mu_B * target_norms[n] * np.dot(Beff_n, angles_der2)

        return grad

    def minimize_energy(self, t, M_init, solver_kwargs=None):
        """
        Minimize total energy at time t using scipy.optimize.minimize with fixed per-spin magnitudes.
        
        Args:
            t (float): Time at which energy/fields are evaluated.
            M_init (np.ndarray): Initial configuration, shape (n_mag, 3).
            solver_kwargs (dict|None): Keyword arguments forwarded to scipy.optimize.minimize.
                Recognized entries include 'method', 'jac', 'hess', 'bounds', 'tol', 'callback', 'options', etc.
                If 'method' is not provided, defaults to 'SLSQP'.
        
        Returns:
            tuple: (M_min, energy_dict) where M_min has shape (n_mag, 3), and energy_dict is from energy().
        
        Notes:
            - Per-spin norms are enforced via equality constraints |M_i|^2 = |M_i(0)|^2.
            - After optimization, norms are checked and an error is raised if they differ,
              rather than silently rescaling the solution.
        """

        M0 = np.asarray(M_init, dtype=float)
        if M0.shape != (self.n_mag, 3):
            raise ValueError(f"M_init shape must be ({self.n_mag}, 3), but got {M0.shape}")
        target_norms = np.linalg.norm(M0, axis=1)
        # Objective: total energy
        def objective(m_flat):
            M = m_flat.reshape((self.n_mag, 3))
            return float(self.energy(t, M)['total'])
        # Always enforce |M_i|^2 = |M_i(0)|^2 for each spin i
        constraints = []
        for i in range(self.n_mag):
            r2 = float(target_norms[i] ** 2)
            def make_con(i, r2):
                return lambda m_flat, i=i, r2=r2: np.dot(m_flat[3*i:3*i+3], m_flat[3*i:3*i+3]) - r2
            constraints.append({'type': 'eq', 'fun': make_con(i, r2)})
        # Prepare solver kwargs
        kw = dict(solver_kwargs) if isinstance(solver_kwargs, dict) else {}
        method = kw.pop('method', 'SLSQP')
        res = scipy.optimize.minimize(
            objective,
            M0.flatten(),
            method=method,
            constraints=constraints,
            **kw
        )
        M_opt = res.x.reshape((self.n_mag, 3))
        # Verify magnitudes; do not rescale automatically
        final_norms = np.linalg.norm(M_opt, axis=1)
        if not np.allclose(final_norms, target_norms, rtol=1e-6, atol=1e-12):
            raise RuntimeError(
                "Optimizer returned spins with norms differing from the constraints. "
                f"Initial norms: {target_norms}, final norms: {final_norms}. "
                "Consider tightening solver tolerances (e.g., lower 'tol', increase 'maxiter')."
            )
        return M_opt, self.energy(t, M_opt), res

    def minimize_energy_angles(self, t, M_init, solver_kwargs=None):
        """
        Minimize total energy at time t using scipy.optimize.minimize over spherical angles.

        Each spin is parameterized by its polar angle theta and azimuthal angle phi while
        keeping |M_i| fixed to its initial magnitude. This formulation supplies an analytic
        gradient, which can improve convergence for some optimizers.
        """

        M0 = np.asarray(M_init, dtype=float)
        if M0.shape != (self.n_mag, 3):
            raise ValueError(f"M_init shape must be ({self.n_mag}, 3), but got {M0.shape}")
        target_norms = np.linalg.norm(M0, axis=1)
        if not np.all(target_norms > 0.0):
            raise ValueError("All initial spin norms must be non-zero.")
        angles0 = _M_to_angles(M0, target_norms, self.n_mag)

        def objective_angles(angles_flat):
            M = _angles_to_M(angles_flat, target_norms, self.n_mag)
            return float(self.energy(t, M)['total'])

        def gradient_angles_local(angles_flat):
            return self.gradient_angles(t, angles_flat, target_norms)

        # Bounds for each (theta, phi) pair
        default_bounds = []
        for _ in range(self.n_mag):
            default_bounds.append((0.0, np.pi))     # theta
            default_bounds.append((-np.pi, np.pi))  # phi

        kw = dict(solver_kwargs) if isinstance(solver_kwargs, dict) else {}
        method = kw.pop('method', 'SLSQP')
        bounds = kw.pop('bounds', default_bounds)
        jac = kw.pop('jac', gradient_angles_local)

        res = scipy.optimize.minimize(
            objective_angles,
            angles0,
            method=method,
            jac=jac,
            bounds=bounds,
            **kw
        )
        M_opt = _angles_to_M(res.x, target_norms, self.n_mag)
        final_norms = np.linalg.norm(M_opt, axis=1)
        if not np.allclose(final_norms, target_norms, rtol=1e-6, atol=1e-12):
            raise RuntimeError(
                "Optimizer returned spins with norms differing from the constraints. "
                f"Initial norms: {target_norms}, final norms: {final_norms}. "
                "Consider tightening solver tolerances (e.g., lower 'tol', increase 'maxiter')."
            )
        return M_opt, self.energy(t, M_opt), res

    def global_minimize_energy(self, t, M_init, solver_kwargs=None, dedup_tol=1e-4, use_gradient=False):
        """
        Global minimization of total energy at time t using scipy.optimize.shgo,
        parameterizing each spin by spherical angles (theta, phi) with fixed radii.

        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo
        
        Args:
            t (float): Time at which energy/fields are evaluated.
            M_init (np.ndarray): Initial configuration, shape (n_mag, 3). Sets the per-spin magnitudes.
                Otherwise, M_init is irrelevant.
            solver_kwargs (dict|None): Keyword arguments forwarded to scipy.optimize.shgo
                (e.g., 'iter', 'sampling_method', 'options', etc.).
            dedup_tol (float|None): Frobenius-norm tolerance to treat two M configurations as identical.
                Must be >= 0 when provided. Defaults to 1e-4. Set smaller to keep more near-duplicates.
                Set to None to disable deduplication.
            use_gradient (bool): If True, supply the analytic gradient of the energy with respect
                to the angle parameters to SHGO. Defaults to False.
        
        Returns:
            tuple: (M_minima_list, energies_list, res)
                - M_minima_list: list of np.ndarray each of shape (n_mag, 3) for every local minimum found
                - energies_list: list of floats with the corresponding energies
                - res: the full SHGO result object from SciPy
        
        Notes:
            - Radii |M_i| are fixed to |M_i(0)| by optimizing only angles.
            - Bounds are simple: theta in [0, pi], phi in [-pi, pi] for each spin.
            - After optimization, norms are verified and an error is raised if they differ.
        """

        M0 = np.asarray(M_init, dtype=float)
        if M0.shape != (self.n_mag, 3):
            raise ValueError(f"M_init shape must be ({self.n_mag}, 3), but got {M0.shape}")
        target_norms = np.linalg.norm(M0, axis=1)
        if not np.all(target_norms > 0.0):
            raise ValueError("All initial spin norms must be non-zero.")
        # Bounds: theta in [0, pi], phi in [-pi, pi] for each spin
        bounds = []
        for _ in range(self.n_mag):
            bounds.append((0.0, np.pi))     # theta
            bounds.append((-np.pi, np.pi))  # phi
        # Prepare kwargs
        kw = dict(solver_kwargs) if isinstance(solver_kwargs, dict) else {}
        kw.setdefault('sampling_method', 'sobol')
        kw.setdefault('n', 1000)
        kw.setdefault('iters', 2)
        print(kw)
        def objective_angles(angles_flat):
            M = _angles_to_M(angles_flat, target_norms, self.n_mag)
            return float(self.energy(t, M)['total'])

        def gradient_angles(angles_flat):
            return self.gradient_angles(t, angles_flat, target_norms)

        # Objective and optional gradient in angle space
        options = kw.setdefault('options', {})
        if use_gradient and 'jac' not in options:
            options['jac'] = gradient_angles

        res = scipy.optimize.shgo(objective_angles, bounds=bounds, **kw)
        # Collect all minima in angle space
        angles_min_all = np.atleast_2d(res.xl) if hasattr(res, 'xl') else np.atleast_2d(res.x)
        M_min_list = []
        E_min_list = []
        for ang in angles_min_all:
            M_i = _angles_to_M(ang, target_norms, self.n_mag)
            M_min_list.append(M_i)
            E_min_list.append(float(self.energy(t, M_i)['total']))
        # Deduplicate near-identical minima by Frobenius norm of differences
        if dedup_tol is None:
            return M_min_list, E_min_list, res
        if dedup_tol < 0:
            raise ValueError("dedup_tol must be non-negative or None.")
        order = np.argsort(E_min_list)
        unique_M = []
        unique_E = []
        for idx in order:
            cand_M = M_min_list[idx]
            is_duplicate = False
            for kept_M in unique_M:
                if np.linalg.norm(cand_M - kept_M) <= float(dedup_tol):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_M.append(cand_M)
                unique_E.append(E_min_list[idx])
        return unique_M, unique_E, res

    def parallel_minimize_energy(
        self,
        t,
        starting_points,
        solver_kwargs=None,
        processes=None,
        dedup_atol=1e-6,
        mode='minimize_angles',
        llg_kwargs=None,
    ):
        """
        Runs multiple local minimizations or LLG relaxations from user-provided starting 
        configurations in parallel processes, deduplicates the resulting minima, and returns 
        them sorted by total energy.

        Args:
            t (float): Time at which the energy is evaluated.
            starting_points (Sequence[np.ndarray]): Iterable of initial configurations. Each entry
                must have shape (n_mag, 3) and non-zero per-spin magnitudes.
            solver_kwargs (dict|None): Passed verbatim to the minimization function for every start.
            processes (int|None): Number of worker processes. Defaults to None (executor chooses).
                Set to 1 to force sequential execution.
            dedup_atol (float|None): Absolute tolerance passed to np.allclose when comparing two
                minima. Must be non-negative. Set to None to disable deduplication.
            mode (str): Minimization mode. Options:
                - 'minimize_angles' (default): Use minimize_energy_angles (spherical coordinates)
                - 'minimize': Use minimize_energy (Cartesian coordinates with constraints)
                - 'llg': Run solve_LLG dynamics from each starting point
            llg_kwargs (dict|None): Additional keyword arguments forwarded to solve_LLG for
                mode='llg'. Must include 'tf'. Optional keys such as 't0', 'ncp', or
                'solver_kwargs' are also supported.

        Returns:
            tuple:
                - minima (list[np.ndarray]): Unique minima (shape (n_mag, 3)), sorted by total energy.
                - energies (list[float]): Total energies corresponding to each minimum.
                - optimizer_results (list[Any]): Either SciPy OptimizeResult objects (mode='minimize'
                  or 'minimize_angles') or Solution objects returned by solve_LLG (mode='llg').
                - start_indices (list[int]): Indices of the originating starting points.
        """
        if starting_points is None:
            raise ValueError("starting_points must be a non-empty sequence of initial configurations.")
        start_list = [np.asarray(sp, dtype=float) for sp in starting_points]
        if len(start_list) == 0:
            raise ValueError("starting_points must contain at least one configuration.")
        for idx, start in enumerate(start_list):
            if start.shape != (self.n_mag, 3):
                raise ValueError(
                    f"Starting point at index {idx} has shape {start.shape}, expected ({self.n_mag}, 3)."
                )
            norms = np.linalg.norm(start, axis=1)
            if np.any(norms <= 0.0):
                raise ValueError(f"All spins must have non-zero magnitude (issue at start index {idx}).")

        if dedup_atol is not None:
            dedup_atol = float(dedup_atol)
            if dedup_atol < 0.0:
                raise ValueError("dedup_atol must be non-negative or None.")

        mode_normalized = str(mode).lower()
        if mode_normalized not in ('minimize', 'minimize_angles', 'llg'):
            raise ValueError("mode must be 'minimize', 'minimize_angles', or 'llg'.")
        if mode_normalized == 'llg':
            if not isinstance(llg_kwargs, dict):
                raise ValueError("llg_kwargs must be provided as a dict when mode='llg'.")
            if 'tf' not in llg_kwargs:
                raise ValueError("llg_kwargs must include 'tf' when mode='llg'.")

        def _run_single(start):
            return _run_parallel_task(self, mode_normalized, t, start, solver_kwargs, llg_kwargs)

        results = []
        run_parallel = len(start_list) > 1 and (processes is None or processes > 1)
        if run_parallel:
            
            payloads = [
                (self, mode_normalized, t, start, solver_kwargs, llg_kwargs, i)
                for i, start in enumerate(start_list)
            ]
            with _MultiprocessPool(processes=processes) as pool:
                # Use imap_unordered to get results as they complete
                results_with_idx = []
                with tqdm(total=len(payloads), desc="Minimizing", unit="config") as pbar:
                    for result in pool.imap_unordered(_parallel_task_worker, payloads):
                        results_with_idx.append(result)
                        pbar.update(1)
            # Unpack
            # results_with_idx is list of (idx, M_opt, energy, res)
            # Sort by idx just in case, though map preserves order
            results_with_idx.sort(key=lambda x: x[0])
            results = [x[1:] for x in results_with_idx]
        else:
            # Sequential with progress bar
            for start in tqdm(start_list, desc="Minimizing", unit="config"):
                results.append(_run_single(start))

        # results is list of (M_opt, energy, aux_res)
        # Sort by energy
        # We want to return lists of minima, energies, results, and original indices
        # So let's attach original index if not already tracked.
        # Actually we want to deduplicate first?
        # The docstring says "deduplicates the resulting minima, and returns them sorted by total energy."
        
        # Let's structure data as (energy, M_opt, aux_res, original_idx)
        data = []
        for i, (M_opt, en, aux) in enumerate(results):
            data.append({
                'energy': en,
                'M': M_opt,
                'res': aux,
                'idx': i
            })
        
        # Sort by energy
        data.sort(key=lambda x: x['energy'])

        unique_data = []
        if dedup_atol is None:
            unique_data = data
        else:
            for item in data:
                cand_M = item['M']
                is_dup = False
                for kept in unique_data:
                    if np.allclose(cand_M, kept['M'], atol=dedup_atol):
                        is_dup = True
                        break
                if not is_dup:
                    unique_data.append(item)

        return (
            [d['M'] for d in unique_data],
            [d['energy'] for d in unique_data],
            [d['res'] for d in unique_data],
            [d['idx'] for d in unique_data]
        )
