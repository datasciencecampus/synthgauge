"""Functions for creating toy datasets."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def _adjust_data_elements(data, labels, noise, nan_prop, seed):
    """Adjust the given data and put it into a dataframe.
    This function is not intended to be used directly by users.

    Parameters
    ----------
    data : numpy.ndarray
        The data array to be adjusted.
    labels : numpy.ndarray
        A set of labels for classifying the rows of `data`.
    noise : float
        The amount of noise to inject into the data. Specifically,
        this controls the `scale` parameter of a zero-centred normal
        distribution.
    nan_prop : float
        The proportion of elements to replace with missing values.
    seed : int
        A random seed used to choose missing element indices and sample
        noise.

    Returns
    -------
    pandas.DataFrame
        The adjusted, combined dataframe.
    """

    rng = np.random.default_rng(seed)

    data = np.column_stack((data, labels))

    num_cols = data.shape[1]
    num_nans = int(data.size * nan_prop)
    nan_idxs = rng.choice(data.size, num_nans, replace=False)
    nan_coords = [(idx // num_cols, idx % num_cols) for idx in nan_idxs]

    if nan_coords:
        data[tuple(np.transpose(nan_coords))] = np.nan

    data = pd.DataFrame(data + rng.normal(scale=noise, size=data.shape))

    return data


def make_blood_types_df(noise=0, nan_prop=0, seed=None):
    """Create a toy dataset about blood types and physical atrtibutes.

    This function is used to create data for the package's examples and
    its tests. Its outputs are not intended to imply or be used for any
    meaningful data analysis.

    Parameters
    ----------
    noise : float
        Standard deviation of the Gaussian noise added to the data.
        Default is zero (no noise) and must be non-negative.
    nan_prop : float, default 0
        Proportion of dataset to replace with missing values.
    seed : int, optional
        Seed used by all random samplers. Used for reproducibility.

    Returns
    -------
    data : pandas.DataFrame
        A toy "blood type" dataset.

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
        random_state=seed,
    )

    df = _adjust_data_elements(X, y, noise, nan_prop, seed)

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
