import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
import os
import sys
PARENT = [os.path.join('..')]
sys.path.extend(PARENT)
from dummyestimator import DummyEstimator


class TestDummyEstimator():
    """
    Test for DummyEstimator class
    """

    def test_init(self):
        prob = 0.5
        clf = DummyEstimator(prob)
        assert prob == clf.prob

    def test_bad_init_zero(self):
        prob = 0
        with pytest.raises(Exception):
            clf = DummyEstimator(prob)

    def test_bad_init_one(self):
        prob = 1
        with pytest.raises(Exception):
            clf = DummyEstimator(prob)

    def test_fit(self):
        X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
        y = np.array([1, 2, 3])

        clf = DummyEstimator(1 / 3)
        clf.fit(X, y)

        assert X.shape[1] == clf.coef_.shape[0]
        assert set(clf.classes_) == set(np.unique(y))

    def test_fit_constant(self):
        X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
        y = np.array([1, 1, 1])

        clf = DummyEstimator(0.999)
        clf.fit(X, y)

        assert X.shape[1] == clf.coef_.shape[0]
        assert set(clf.classes_) == set(np.unique(y))

    def test_predict_proba(self):
        fixed_prob = 0.999
        clf = DummyEstimator(fixed_prob)
        X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
        y = np.array([1, 1, 1])
        clf.fit(X, y)
        prob = clf.predict_proba(np.array([[0, 1, 0, 0]]))
        assert fixed_prob == prob

    def test_predict_log_proba(self):
        fixed_prob = 0.999
        clf = DummyEstimator(fixed_prob)
        X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
        y = np.array([1, 1, 1])
        clf.fit(X, y)
        logprob = clf.predict_log_proba(np.array([[0, 1, 0, 0]]))
        assert logprob == np.log(fixed_prob)

    def test_predict_not_fitted(self):
        fixed_prob = 0.999
        clf = DummyEstimator(fixed_prob)
        X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
        with pytest.raises(NotFittedError):
            _ = clf.predict_proba(X)
