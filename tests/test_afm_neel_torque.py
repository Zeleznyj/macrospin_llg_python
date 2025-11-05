import os
import numpy as np
import pytest

from macrospin_llg import Model


def test_afm_neel_fieldlike_switching():
    # Antiferromagnetic Field-like Néel order torque switching test (converted from MATLAB)

    m = Model(2)

    # Exchange across all pairs (MATLAB used 0,0 to denote all; here -1 means all)
    m.add_exchange(-1, -1, 0.1)

    # Cubic anisotropy on all moments
    m.add_ani4(-1, 1e-5)

    # Opposite y-directed fields on the two sublattices
    Bmag = 0.1
    m.add_B(0, np.array([0.0, Bmag, 0.0]))
    m.add_B(1, np.array([0.0, -Bmag, 0.0]))

    # Gilbert damping
    m.ag = 0.005

    # Initial state: [3,0,0] and [-3,0,0]
    M0 = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])

    # Increase solver accuracy (relprec ~ llg_params.relprec in MATLAB)
    sol = m.solve_LLG(tf=0.02, M0=M0, relprec=1e-6)

    # Final-state norm checks against [0,3,0] and [0,-3,0]
    M1_end = sol.M[-1, 0, :]
    M2_end = sol.M[-1, 1, :]
    norm_M1 = np.linalg.norm(M1_end - np.array([0.0, 3.0, 0.0]))
    norm_M2 = np.linalg.norm(M2_end - np.array([0.0, -3.0, 0.0]))

    # Relative errors are norm/3 in the original printouts; assertions on absolute norms
    assert norm_M1 < 1e-4, f"M1 norm test failed: {norm_M1}"
    assert norm_M2 < 1e-4, f"M2 norm test failed: {norm_M2}"

    # Full-trajectory comparison against MATLAB reference stored in sol_Neel_new.mat
    # File location can be overridden by NEEL_MAT_FILE; default to repo root
    try:
        from scipy.io import loadmat
    except Exception as exc:
        pytest.skip(f"SciPy required to load MATLAB reference: {exc}")

    default_mat_path = "sol_Neel_new.mat"
    if not os.path.isfile(default_mat_path):
        # Also try project root
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sol_Neel_new.mat"))
        if os.path.isfile(alt_path):
            default_mat_path = alt_path
        else:
            pytest.skip(f"Reference MATLAB file not found: {default_mat_path}")

    mat = loadmat(default_mat_path, squeeze_me=True)
    if "T" not in mat or "M" not in mat:
        pytest.skip("MATLAB file must contain variables 'T' and 'M'")

    T_ref = np.asarray(mat["T"]).astype(float).ravel()
    M_ref_flat = np.asarray(mat["M"]).astype(float).ravel()

    assert M_ref_flat.size % 6 == 0, "Length of M in MAT file is not divisible by 6"
    nt_ref = M_ref_flat.size // 6

    # Reshape MATLAB-flattened (column-major) sequence into (nt, 6)
    M_ref6 = np.reshape(M_ref_flat, (6, nt_ref), order="F").T  # (nt, 6)
    # Split into moments: columns 0:3 -> M1, 3:6 -> M2
    M_ref = np.stack([M_ref6[:, 0:3], M_ref6[:, 3:6]], axis=1)  # (nt, 2, 3)

    # Align solver output to the same time grid as MATLAB reference
    tf_ref = float(T_ref[-1])
    ncp_ref = int(len(T_ref) - 1)

    sol_refgrid = m.solve_LLG(tf=tf_ref, M0=M0, relprec=1e-6, ncp=ncp_ref)

    # If times differ slightly, ensure lengths match and interpolate our solution onto T_ref
    if sol_refgrid.t.shape != T_ref.shape or not np.allclose(sol_refgrid.t, T_ref, rtol=1e-9, atol=1e-12):
        # Linear interpolation per component
        M_interp = np.empty((len(T_ref), 2, 3), dtype=float)
        for a in range(2):
            for j in range(3):
                M_interp[:, a, j] = np.interp(T_ref, sol_refgrid.t, sol_refgrid.M[:, a, j])
        M_model = M_interp
    else:
        M_model = sol_refgrid.M

    # Compute relative Frobenius norm of trajectory difference
    diff = M_model - M_ref
    denom = max(np.linalg.norm(M_ref), 1e-15)
    rel_err = np.linalg.norm(diff) / denom

    assert rel_err < 1e-3, f"Full-trajectory relative error too large: {rel_err}"


