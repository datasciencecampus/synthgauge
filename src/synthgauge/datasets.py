import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_blood_types_df(noise=0, proportion_nan=0, random_seed=42):
    """Create Dummy Data for examples

    This function creates dummy data for the synthgauge examples.

    Parameters
    ----------
    noise : int
        Standard deviation of the Gaussian noise to add to the data, default
        zero.
    proportion_nan: float [0,1]
        Proportion of dataset to replace with nans.
    random_seed : int
        Use for reproducibility.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    The amout of noise can be tuned to crudely simulate the creation of
    synthetic data.
    """

    X, y = make_classification(n_samples=1000,
                               n_features=5,
                               n_informative=3,
                               n_redundant=2, n_classes=4,
                               weights=[0.4, 0.3, 0.2, 0.1],
                               flip_y=0.1,
                               random_state=random_seed)

    np.random.seed(random_seed)
    random.seed(random_seed)

    mat = np.column_stack((X, 3*y))
    prop = int(mat.size * proportion_nan)
    i = [random.choice(range(mat.shape[0])) for _ in range(prop)]
    j = [random.choice(range(mat.shape[1])) for _ in range(prop)]
    mat[i, j] = np.NaN

    df = pd.DataFrame(mat + np.random.normal(scale=noise, size=(1000, 6)))

    df.columns = ['age', 'height', 'weight',
                  'hair_colour', 'eye_colour', 'blood_type']

    df.age = np.abs(round(df.age*(52/9)+44))
    df.height = np.abs(round(df.height*(52/9)+175))
    df.weight = np.abs(round(df.weight*(52/9)+80))
    df.hair_colour = (pd.cut(df.hair_colour, 4, labels=[
        'Red', 'Black', 'Brown', 'Blonde'])
        .cat.as_unordered())
    df.eye_colour = pd.cut(df.eye_colour, 3, labels=[
                           'Blue', 'Brown', 'Green']).cat.as_unordered()
    df.blood_type = pd.cut(df.blood_type, 4, labels=[
                           'O', 'A', 'B', 'AB']).cat.as_unordered()

    return df


if __name__ == '__main__':
    pass
