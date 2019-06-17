"""
Helper functions
Exists to prevent circular dependencies
"""

import numpy as np


def round_labels(labels):
    """
    Round the labels for doing hard expectation-maximization.
    The labels are rounded up to 1 if greater than 0.5, and down to 0 if below
    0.5. In the (unlikely) event they are exactly equal to 0.5, it is randomly
    rounded up or down.

    Arguments
    ---------
    labels: array-like, values of the hidden variables

    Returns
    ---------
    labels: array-like, rounded values of the hidden variables

    """
    labels = [np.random.choice([0, 1]) if x ==
              0.5 else 1 if x > 0.5 else 0 for x in labels]
    return labels
