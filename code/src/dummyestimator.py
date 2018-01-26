"""
"""

import warnings
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_is_fitted


class DummyEstimator(BaseEstimator, ClassifierMixin):
    """
    Dummy estimator which always predict a given probability regardless of the
    input data.
    Useful for creating models for constant input.
    Only used when encountering constant columns under hard EM!

    Parameters
    ----------
    prob: float, probability to be returned regardless of the input

    Attributes
    ----------
    coef_: array, (1, n_features)
    classes_: array, classes found in the dataset

    """

    def __init__(self, prob):
        self.prob = prob

    def fit(self, X, y):
        """
        """
        X, y = check_X_y(X, y)

        self.coef_ = np.zeros(X.shape[1])
        self.classes_ = np.unique(y)

        return self

    def predict_proba(self, X):
        """
        """
        if not hasattr(self, 'coef_'):
            raise NotFittedError('Fit model before predicting')
        proba = np.array([self.prob] * X.shape[0])
        proba = np.reshape(proba, (X.shape[0], 1))
        return proba

    def predict_log_proba(self, X):
        """
        """
        return np.log(self.predict_proba(X))
