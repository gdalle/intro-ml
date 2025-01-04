import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""# Machine Learning in 1h - practice""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    To display this notebook, go to ...

    Then you can edit its cells and run them entirely in your browser.
    """
    ).callout(kind="info")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import sklearn, sklearn.datasets
    return np, pd, plt, sklearn


@app.cell
def _(mo):
    mo.md(r"""## The Titanic dataset""")
    return


@app.cell
def _(sklearn):
    titanic = sklearn.datasets.fetch_openml(data_id=40945)
    return (titanic,)


@app.cell
def _(titanic):
    titanic.data
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
