""" Functions for creating toy datasets. """

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_blood_types_df(noise=0, proportion_nan=0, random_seed=None):
    """Create a toy dataset about blood types and physical atrtibutes.

    This function is used to create data for the package's examples and
    its tests. Its outputs are not intended to imply or be used for any
    meaningful data analysis.

    Parameters
    ----------
    noise : float
        Standard deviation of the Gaussian noise to add to the data.
        Default is one and must be non-negative.
    proportion_nan: float [0, 1]
        Proportion of dataset to replace with missing values. Default is
        zero, i.e. complete data.
    random_seed : int
        The seed used for any PRNGs. Used for reproducibility.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    The amount of noise can be tuned to crudely simulate the creation of
    synthetic data.
    """

    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_classes=4,
        weights=[0.4, 0.3, 0.2, 0.1],
        flip_y=0.1,
        random_state=random_seed,
    )

    rng = np.random.default_rng(random_seed)

    mat = np.column_stack((X, 3 * y))

    num_cols = mat.shape[1]
    num_nans = int(mat.size * proportion_nan)
    nan_idxs = rng.choice(mat.size, num_nans, replace=False)
    nan_coords = [(idx // num_cols, idx % num_cols) for idx in nan_idxs]

    for x, y in nan_coords:
        mat[x, y] = np.nan

    df = pd.DataFrame(mat + rng.normal(scale=noise, size=(1000, 6)))

    df.columns = [
        "age",
        "height",
        "weight",
        "hair_colour",
        "eye_colour",
        "blood_type",
    ]

    df.age = np.abs(round(df.age * (52 / 9) + 44))
    df.height = np.abs(round(df.height * (52 / 9) + 175))
    df.weight = np.abs(round(df.weight * (52 / 9) + 80))
    df.hair_colour = pd.cut(
        df.hair_colour, 4, labels=["Red", "Black", "Brown", "Blonde"]
    ).cat.as_unordered()
    df.eye_colour = pd.cut(
        df.eye_colour, 3, labels=["Blue", "Brown", "Green"]
    ).cat.as_unordered()
    df.blood_type = pd.cut(
        df.blood_type, 4, labels=["O", "A", "B", "AB"]
    ).cat.as_unordered()

    return df


if __name__ == "__main__":
    pass
