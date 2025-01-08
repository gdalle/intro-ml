import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium", app_title="ML in 1h - practice")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Machine Learning in 1h - practice""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This is a `marimo` notebook, which works differently from `jupyter` (hopefully in a more intuitive way).

    If you are confused, start by reading this tutorial: https://docs.marimo.io/guides/reactivity/""").callout()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    return np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Titanic dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We retrieve it from OpenML, which contains lots of other datasets (https://www.openml.org/search?type=data&status=active&id=40945)""")
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    titanic = fetch_openml(data_id=40945)
    return fetch_openml, titanic


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The meanings of column names are described at https://www.kaggle.com/c/titanic/data

        Which ones do you think will be relevant to predict passenger survival?
        """
    )
    return


@app.cell
def _(titanic):
    titanic.feature_names
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Let's start by looking at our data""")
    return


@app.cell
def _(titanic):
    data, target = titanic.data, titanic.target.astype(int)
    return data, target


@app.cell
def _(data):
    data
    return


@app.cell
def _(target):
    target
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also ask for summary statistics""")
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell
def _(data):
    data.nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And finally, let's try some plots after adding the survival column""")
    return


@app.cell
def _(data, pd, target):
    total_data = pd.concat([data, target], axis=1)
    return (total_data,)


@app.cell
def _(sns, total_data):
    sns.pairplot(total_data, hue="survived")
    return


@app.cell
def _(sns, total_data):
    sns.catplot(total_data, x="age", y="sex", hue="survived", alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Pipeline components""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Select and standardize features for prediction (https://scikit-learn.org/stable/modules/preprocessing.html)""")
    return


@app.cell
def _():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    transformer = make_column_transformer(
        (
            StandardScaler(),
            [
                "age",
                "sibsp",
                "parch",
                "fare",
            ],
        ),
        (
            OneHotEncoder(),
            [
                "pclass",
                "sex",
            ],
        ),
        remainder="drop",
    )
    return (
        OneHotEncoder,
        StandardScaler,
        make_column_transformer,
        transformer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Impute missing data (https://scikit-learn.org/stable/modules/impute.html)""")
    return


@app.cell
def _():
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="most_frequent")
    return SimpleImputer, imputer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Choose the simplest classifier (https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)""")
    return


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    return LogisticRegression, classifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fitting and prediction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Split data into train and test set""")
    return


@app.cell
def _(data, target):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, train_size=0.6, random_state=0
    )
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell
def _(X_test, X_train):
    X_train.shape, X_test.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Gather all steps into a pipeline (https://scikit-learn.org/stable/modules/compose.html) and `fit`.""")
    return


@app.cell
def _(X_train, classifier, imputer, transformer, y_train):
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(transformer, imputer, classifier)
    pipe.fit(
        X_train, y_train
    )  # always modify the pipeline in the same cell where it is created (marimo quirk)
    return make_pipeline, pipe


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Model tuning and selection""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To select models or tune hyperparameters, cross-validation on the training set is a way to avoid a validation set (https://scikit-learn.org/stable/modules/cross_validation.html)""")
    return


@app.cell
def _(X_train, pipe, y_train):
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    return cross_val_score, scores


@app.cell
def _(scores):
    scores
    return


@app.cell
def _(scores):
    scores.mean()
    return


@app.cell(hide_code=True)
def _(mo, scores):
    mo.md(f"""### Challenge

    Try modifying the pipeline to increase the cross-validated score, currently at **{scores.mean()}**.

    Some ideas:

    - choose a different classifier
    - modify the list of features
    - change missing value imputation
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Final evaluation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Final evaluation is always on the test set we held out""")
    return


@app.cell
def _(X_test, pipe, y_test):
    pipe.score(X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The default metric for our classifier is accuracy but choosing the right metric is a nontrivial task (https://scikit-learn.org/stable/modules/model_evaluation.html)""")
    return


if __name__ == "__main__":
    app.run()
