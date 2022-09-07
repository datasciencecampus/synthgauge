"""Functions for the generic measures from the 2018 NIST competition."""


def three_way_marginals(real, synth, seed=None):
    """The first generic measure based on similarity of 3-way marginals.

    In essence, calculate the summed absolute deviation between the real
    and synthetic data across an array of randomly sampled 3-way
    marginals. Transform and summarise these deviations to give a single
    score between 0 and 1.

    Details can be found at https://doi.org/10.6028/NIST.TN.2151.
    """
