import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Machine Learning in 1h - theory""")
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
    """
    ).callout(kind="info")
    return


@app.cell
def _(mo):
    mo.md("""## Key concepts""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Can machines learn?

        * Artificial Intelligence
        * Machine Learning
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Types of learning

        * Supervised
        * Unsupervised
        * Reinforcement
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Types of labels

        * Classification (categorical label)
        * Regression (quantitative label)
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""## A few algorithms""")
    return


@app.cell
def _(mo):
    mo.md("""### Linear regression""")
    return


@app.cell
def _(mo):
    mo.md("""### Logistic regression""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Before and after""")
    return


@app.cell
def _(mo):
    mo.md("""### Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md("""### Performance evaluation""")
    return


@app.cell
def _(mo):
    mo.md("""### Metrics""")
    return


@app.cell
def _(mo):
    mo.md("""## Software tools""")
    return


@app.cell
def _(mo):
    mo.md("""### `scikit-learn`""")
    return


@app.cell
def _(mo):
    mo.md("""### Choosing an algorithm""")
    return


@app.cell
def _(mo):
    mo.md("""## Going further""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Books

        * [The Hundred-page Machine Learning Book](https://themlbook.com/) & [Machine Learning Engineering](http://mlebook.com/) (Burkov, 2019 & 2020)
        * [Machine Learning with PyTorch and Scikit-Learn](https://sebastianraschka.com/blog/2022/ml-pytorch-book.html) (Raschka, 2022)
        * [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) (Russell and Norvig, 2021)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Courses

        * [Google Machine Learning Education](https://developers.google.com/machine-learning)
        * [Coursera Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
