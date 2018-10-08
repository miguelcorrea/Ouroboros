"""
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.exceptions import NotFittedError


class DummyEstimator(BaseEstimator, ClassifierMixin):
    """
    Dummy estimator which always predict a given probability regardless of the
    input data.
    As it subclasses sklearn classifiers, it can be easily used
    without distinguishing it from the SGDClassifier we normally use.
    This is useful (and only used) when a constant MSA column is encountered in
    hard EM. This can happen if all the putatively interacting examples have an
    the same residue in a given column. We can then give a fixed (high)
    probability for that residue. All others will get a pseudocount
    (not implemented here).

    Parameters
    ----------
    prob: float, probability to be returned regardless of the input

    Attributes
    ----------
    coef_: array, (1, n_features)
    classes_: array, classes found in the dataset
    """

    def __init__(self, prob):
        if 0 < prob < 1:
            self.prob = prob
        else:
            raise ValueError("Probability must be within 0 and 1")

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
