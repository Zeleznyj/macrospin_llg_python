import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    from macrospin_llg import Model
    import numpy as np
    return Model, np


@app.cell
def _(Model, np):
    m = Model(2)

    m.add_exchange(-1, -1, 0.1)

    m.add_ani4(-1, 1e-5)

    Bmag = 0.1
    m.add_B(0, np.array([0.0, Bmag, 0.0]))
    m.add_B(1, np.array([0.0, -Bmag, 0.0]))

    m.ag = 0.005

    M0 = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])

    # Increase solver accuracy (relprec ~ llg_params.relprec in MATLAB)
    sol = m.solve_LLG(tf=0.02, M0=M0, atol=1e-12, rtol=1e-12)
    return (sol,)


@app.cell
def _(sol):
    sol.plot()
    return


@app.cell
def _(sol):
    sol.save('AFM_Neel_ref.npz')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
