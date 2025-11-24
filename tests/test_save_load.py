
import numpy as np
import os
import tempfile
from macrospin_llg.model import Model
from macrospin_llg.solution import Solution

def test_save_load():
    # 1. Setup Model
    n_mag = 2
    model = Model(n_mag)
    model.ag = 0.1
    
    # Add some interactions
    J_val = 1e-3
    model.add_exchange(0, 1, J_val)
    
    K_val = 1e-4
    A_axis = [0, 0, 1]
    model.add_ani2(0, K_val, A_axis)
    
    B_ext = [0.1, 0, 0]
    model.add_B(1, B_ext)
    
    # 2. Create a dummy solution (no need to actually solve for this test, just check data persistence)
    t = np.linspace(0, 1e-9, 10)
    M = np.random.rand(10, n_mag, 3)
    
    sol = Solution(model, t, M)
    
    # Calculate energy before saving to ensure it's populated
    sol.calculate_energy()
    original_E = sol.E

    # 3. Save
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        filename = tmp.name
    
    try:
        sol.save(filename)
        
        # 4. Load
        loaded_sol = Solution.load(filename)
        
        # 5. Verify
        # Check t and M
        assert np.allclose(sol.t, loaded_sol.t), "Time array mismatch"
        assert np.allclose(sol.M, loaded_sol.M), "Magnetization array mismatch"
        
        # Check that model is None
        assert loaded_sol.model is None, "Model should be None"
        
        # Check Energy
        assert loaded_sol.E is not None, "Energy should be loaded"
        assert len(loaded_sol.E) == len(original_E)
        
        # Compare a few values
        for i in range(len(original_E)):
            for k, v in original_E[i].items():
                assert k in loaded_sol.E[i]
                assert np.isclose(loaded_sol.E[i][k], v), f"Energy mismatch at step {i}, key {k}"
        
        print("Save/Load test passed successfully!")
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_save_load()
