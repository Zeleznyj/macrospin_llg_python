import numpy as np
from scipy.interpolate import interp1d
from macrospin_llg.model import Model
from macrospin_llg.solution import Solution
import os

def test_FM_field():
    # Setup model
    m = Model(1)
    m.add_B(0, [0, 0, 10])
    m.ag = 0.1

    _kwargs = {'atol': 1e-12, 'rtol': 1e-12}
    _M_init = np.array([[0.1, 0.1, -3]])
    
    # Solve
    _sol = m.solve_LLG(0.1, _M_init, **_kwargs)

    # Load reference
    # Assuming FM_field.npz is in the same directory as this test file
    ref_path = os.path.join(os.path.dirname(__file__), "reference/FM_field_ref.npz")
    if not os.path.exists(ref_path):
        raise FileNotFoundError("Reference file not found.")
        
    _sol_ref = Solution.load(ref_path)

    # Interpolate calculated solution to reference times
    # _sol.M shape is (T, N, 3). Interpolate along axis 0 (time).
    # We use linear interpolation as it's robust; cubic might overshoot if points are sparse.
    f_interp = interp1d(_sol.t, _sol.M, axis=0, kind='linear', fill_value="extrapolate")
    M_interp = f_interp(_sol_ref.t)

    # Compare M
    # We compare the interpolated values with the reference values.
    np.testing.assert_allclose(M_interp, _sol_ref.M, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    test_FM_field()