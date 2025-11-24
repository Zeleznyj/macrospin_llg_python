import numpy as np
from scipy.interpolate import interp1d
from macrospin_llg.model import Model
from macrospin_llg.solution import Solution
import os

def test_afm_neel_fieldlike_switching():
    # Setup model
    m = Model(2)
    m.add_exchange(-1, -1, 0.1)
    m.add_ani4(-1, 1e-5)
    
    Bmag = 0.1
    m.add_B(0, np.array([0.0, Bmag, 0.0]))
    m.add_B(1, np.array([0.0, -Bmag, 0.0]))
    
    m.ag = 0.005
    
    M0 = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    
    # Solve
    # Using same tolerances as reference generation
    sol = m.solve_LLG(tf=0.02, M0=M0, atol=1e-12, rtol=1e-12)
    
    # Load reference
    ref_path = os.path.join(os.path.dirname(__file__), "reference/AFM_Neel_ref.npz")
    if not os.path.exists(ref_path):
        raise FileNotFoundError("Reference file not found.")
        
    sol_ref = Solution.load(ref_path)
    
    # Interpolate calculated solution to reference times
    f_interp = interp1d(sol.t, sol.M, axis=0, kind='linear', fill_value="extrapolate")
    M_interp = f_interp(sol_ref.t)
    
    # Compare M
    np.testing.assert_allclose(M_interp, sol_ref.M, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    test_afm_neel_fieldlike_switching()


