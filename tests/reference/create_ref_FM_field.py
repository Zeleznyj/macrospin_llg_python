import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    from macrospin_llg import Model
    return (Model,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(Model):
    m = Model(1)
    m.add_B(0,[0,0,10])
    m.ag = 0.1
    return (m,)


@app.cell
def _(m, np):
    _kwargs = {'atol': 1e-12, 'rtol': 1e-12}
    _M_init = np.array([[0.1,0.1,-3]])
    _sol = m.solve_LLG(0.1,_M_init, **_kwargs)
    _sol.plot()
    _sol.save('FM_field_ref.npz')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
