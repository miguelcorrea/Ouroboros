"""
Unit tests for corrmut module
"""
import numpy as np
from math import isclose
from sklearn.linear_model import SGDClassifier
import inspect
import corrmut
import msa_fun
from globalvars import AA_TABLE


class TestLabelUpdate():
    """
    Class to test the corrmut.update_labels function
    """
    ###########
    # Hard EM #
    ###########

    def test_hard_update_one(self):
        """
        Compare obtained labels to expected values over a range of f_int values
        """

        alt = [0.90, 0.80, 0.75, 0.5, 0.2, 0.1]
        null = [0.30, 0.20, 0.30, 0.40, 0.80, 0.80]
        f_int = [1e-6, 0.1, 0.25, 0.5, 0.75, 0.90, 0.99 - 1e-6]
        expected_z = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]

        for idx, val in enumerate(f_int):
            labels = corrmut.update_labels(alt, null, val, 'hard')
            assert expected_z[idx] == labels

        ###########
        # Soft EM #
        ###########

    def test_soft_update_one(self):
        """
        Compare obtained labels to expected values
        """
        alt = [0.90, 0.80, 0.75, 0.5, 0.2, 0.1]
        null = [0.30, 0.20, 0.30, 0.40, 0.80, 0.80]
        f_int = [1e-6, 0.1, 0.25, 0.5, 0.75, 0.90, 0.99 - 1e-6]
        expected_z = [[1.82211730e-06, 1.82211730e-06, 1.56831129e-06,
                       1.10517080e-06, 5.48811884e-07, 4.96585554e-07],
                      [0.16836988, 0.16836988, 0.1483976, 0.10936687,
                       0.05747434, 0.05229093],
                      [0.37786684, 0.37786684, 0.34330232, 0.26921435,
                       0.1546466, 0.14202007],
                      [0.64565631, 0.64565631, 0.61063923, 0.52497919,
                       0.35434369, 0.33181223],
                      [0.8453534, 0.8453534, 0.82471321,
                          0.76827783, 0.62213316, 0.5983542],
                      [0.94252566, 0.94252566, 0.93383972,
                          0.90864692, 0.83163012, 0.81716017],
                      [0.99448646, 0.99448646, 0.99359989, 0.9909421,
                       0.9819256, 0.9800626]]
        for idx, val in enumerate(f_int):
            labels = corrmut.update_labels(alt, null, val, 'soft')
            assert np.allclose(labels, expected_z[idx], rtol=1e-6)


class TestConvergence():
    """
    Class to test the corrmut.has_converged function
    """

    ###########
    # Hard EM #
    ###########
    def test_hard_conv_1(self):
        labels = [1] * 10
        pre_labels = [1] * 10
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert converged

    def test_hard_conv_2(self):
        labels = [0] * 10
        pre_labels = [0] * 10
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert converged

    def test_hard_conv_3(self):
        labels = [1] * 10
        pre_labels = [0] * 10
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert not converged

    def test_hard_conv_4(self):
        labels = [0] * 10
        pre_labels = [1] * 10
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert not converged

    def test_hard_conv_5(self):
        labels = [1, 1, 1, 0, 0, 0]
        pre_labels = [0, 0, 0, 1, 1, 1]
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert not converged

    def test_hard_conv_6(self):
        labels = [1, 1, 1, 1, 0]
        pre_labels = [1, 1, 1, 1, 1]
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert not converged

    def test_hard_conv_7(self):
        labels = [1, 1, 1, 0, 0, 0]
        pre_labels = [1, 1, 1, 0, 0, 0]
        converged = corrmut.has_converged(labels, pre_labels, "hard")
        assert converged

    ###########
    # Soft EM #
    ###########

    def test_soft_conv_1(self):
        labels = [1] * 10
        pre_labels = [1] * 10
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert converged

    def test_soft_conv_2(self):
        labels = [0] * 10
        pre_labels = [0] * 10
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert converged

    def test_soft_conv_3(self):
        labels = [1] * 10
        pre_labels = [0] * 10
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert not converged

    def test_soft_conv_4(self):
        labels = [0] * 10
        pre_labels = [1] * 10
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert not converged

    def test_soft_conv_5(self):
        labels = [1, 1, 1, 0, 0, 0]
        pre_labels = [0, 0, 0, 1, 1, 1]
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert not converged

    def test_soft_conv_6(self):
        labels = [1, 1, 1, 1, 0]
        pre_labels = [1, 1, 1, 1, 1]
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert not converged

    def test_soft_conv_7(self):
        labels = [1, 1, 1, 0, 0, 0]
        pre_labels = [1, 1, 1, 0, 0, 0]
        converged = corrmut.has_converged(labels, pre_labels, "soft")
        assert converged

    def test_soft_conv_8(self):
        labels = [0.99, 0.95, 0.90, 0.95, 0.1, 0.1, 0.001, 1e-3]
        pre_labels = [[0.99, 0.95, 0.90, 0.95, 0.1, 0.1, 0.001, 1e-16]]
        tol = 5e-3
        converged = corrmut.has_converged(labels, pre_labels, "soft", tol=tol)
        assert converged

    def test_soft_conv_9(self):
        labels = [0.99, 0.95, 0.90, 0.95, 0.1, 0.1, 0.001, 1e-3]
        pre_labels = [[0.99, 0.95, 0.90, 0.95, 0.1, 0.1, 0.001, 1e-32]]
        tol = 5e-3
        converged = corrmut.has_converged(labels, pre_labels, "soft", tol=tol)
        assert converged

    def test_soft_conv_10(self):
        # Difference between two arrays equals tolerance,
        # (would not be enough to reach convergence by itself)
        # but amount of observations must also be taken into account!
        labels = [0.99, 0.99, 0.1, 0.1, 0.005]
        pre_labels = [0.99, 0.99, 0.1, 0.1, 0.01]
        tol = 5e-3
        converged = corrmut.has_converged(labels, pre_labels, "soft", tol=tol)
        assert converged


#####################
# Alternative model #
#####################

class TestGetPosteriorLogProbs():
    """
    Class to test the corrmut.get_posterior_logprobs function
    """

    def test_logprobs_1(self):
        # Tested using sklearn 0.19.2
        # Covariation between two toy alignments
        Y = [3, 11, 3, 11, 3, 11]
        X = np.array([[11, 3, 11, 3, 11, 3]])

        exp_logprobs = np.log([0.79807613, 0.79822048, 0.79807613,
                               0.79822048, 0.79807613, 0.79822048])

        X = X.reshape(6, 1)
        bin_mtx = msa_fun.make_bin_mtx(X, AA_TABLE)
        clf = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.99,
                            n_jobs=2, max_iter=100, random_state=42, alpha=0.1)
        clf.fit(bin_mtx, Y)
        posterior_logprobs = corrmut.get_posterior_logprobs(Y, bin_mtx, clf)

        assert np.allclose(exp_logprobs, posterior_logprobs, rtol=1e-6)

    def test_logprobs_2(self):
        # As before, but introducing testing of pseudocount
        Y = [3, 11, 3, 11, 3, 11, 19]
        X = [11, 3, 11, 3, 11, 3, 20]
        Y_train = Y[:-1]
        X = np.array(X).reshape(7, 1)
        all_mtx = msa_fun.make_bin_mtx(X, AA_TABLE)

        sig = inspect.signature(corrmut.get_posterior_logprobs)
        pc = np.exp(sig.parameters['pc'].default)

        exp_logprobs = np.log([0.79807613, 0.79822048, 0.79807613,
                               0.79822048, 0.79807613, 0.79822048,
                               pc])

        X_train = np.array(X[:-1]).reshape(6, 1)
        train_mtx = msa_fun.make_bin_mtx(X_train, AA_TABLE)

        clf = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.99,
                            n_jobs=2, max_iter=100, random_state=42, alpha=0.1)
        clf.fit(train_mtx, Y_train)
        posterior_logprobs = corrmut.get_posterior_logprobs(Y, all_mtx, clf)

        assert np.allclose(exp_logprobs, posterior_logprobs, rtol=1e-6)


##############
# Null model #
##############

class TestGetNullModel():
    """
    Class to test the corrmut.get_null_model function
    """

    def test_null_one(self):
        # Normal case
        num_mtx = np.array([[1, 2, 3],
                            [1, 3, 3],
                            [3, 2, 5]])
        weights = [0.1, 0.1, 0.9]
        pc_null = 1 / 210

        expected = [{'1': np.log(0.18181818), '3': np.log(1 - 0.18181818)},
                    {'2': np.log(0.90909090), '3': np.log(1 - 0.90909090)},
                    {'3': np.log(0.18181818), '5': np.log(1 - 0.18181818)}]

        null_model = corrmut.get_null_model(num_mtx, weights, pc_null)
        # Check whether probabilities add up to one in all columns
        # Check that calculated probabilities match up with the expected values
        for idx, col in enumerate(null_model):
            assert np.sum(np.exp(list(col.values()))) == 1
            for k in col.keys():
                assert isclose(col[k], expected[idx][k], rel_tol=1e-6)

    def test_null_pseudocount(self):
        # Test including pseudocount
        # Include odd sequence with weight 0
        num_mtx = np.array([[1, 2, 3], [1, 3, 3], [3, 2, 5], [18, 19, 20]])
        pc_null = 1 / 210
        weights = [0.1, 0.1, 0.9, 0.0]

        expected = [{'1': np.log(0.18181818), '3': np.log(1 - 0.18181818),
                     '18': np.log(pc_null)},
                    {'2': np.log(0.90909090), '3': np.log(1 - 0.90909090),
                     '19': np.log(pc_null)},
                    {'3': np.log(0.18181818), '5': np.log(1 - 0.18181818),
                     '20': np.log(pc_null)}]

        null_model = corrmut.get_null_model(num_mtx, weights, pc_null)
        # Check that calculated probabilities match up with the expected values
        for idx, col in enumerate(null_model):
            for k in col.keys():
                assert isclose(col[k], expected[idx][k], rel_tol=1e-6)


class TestScoreNull():
    """
    Class to test the corrmut.score_null function
    """

    def test_null_one(self):
        # Same example data as in test_null_model in TestGetNullModel
        num_mtx = np.array([[1, 2, 3],
                            [1, 3, 3],
                            [3, 2, 5]])
        null_model = [{'1': np.log(0.18181818), '3': np.log(1 - 0.18181818)},
                      {'2': np.log(0.90909090), '3': np.log(1 - 0.90909090)},
                      {'3': np.log(0.18181818), '5': np.log(1 - 0.18181818)}]
        pc_null = 1 / 210

        exp_null_mtx = np.array([[null_model[0]['1'], null_model[1]['2'], null_model[2]['3']],
                                 [null_model[0]['1'], null_model[
                                     1]['3'], null_model[2]['3']],
                                 [null_model[0]['3'], null_model[1]['2'], null_model[2]['5']]])

        null_mtx = corrmut.score_null(num_mtx, null_model, pc_null)
        assert np.allclose(null_mtx, exp_null_mtx, rtol=1e-6)

    def test_null_pseudocount(self):
        # Same data as in test_null_pseudocount
        num_mtx = np.array([[1, 2, 3], [1, 3, 3], [3, 2, 5], [18, 19, 20]])
        pc_null = 1 / 210
        null_model = [{'1': np.log(0.18181818), '3': np.log(1 - 0.18181818), '18': np.log(pc_null)},
                      {'2': np.log(0.90909090), '3': np.log(
                          1 - 0.90909090), '19': np.log(pc_null)},
                      {'3': np.log(0.18181818), '5': np.log(1 - 0.18181818), '20': np.log(pc_null)}]

        exp_null_mtx = np.array([[null_model[0]['1'], null_model[1]['2'], null_model[2]['3']],
                                 [null_model[0]['1'], null_model[
                                     1]['3'], null_model[2]['3']],
                                 [null_model[0]['3'], null_model[
                                     1]['2'], null_model[2]['5']],
                                 [null_model[0]['18'], null_model[1]['19'], null_model[2]['20']]])

        null_mtx = corrmut.score_null(num_mtx, null_model, pc_null)
        assert np.allclose(null_mtx, exp_null_mtx, rtol=1e-6)


###################
# Model selection #
###################

class TestCalcDegreesFreedom():
    """
    Class to test the corrmut.calc_degrees_freedom function
    """

    def test_dfs_one(self):
        # Data simple enough to check the model coefficient matrix by eye
        # Tested using sklearn 0.19.2
        expected_dfs = 6
        # Covariation between two toy alignments, introducing some noise
        Y = [3, 11, 3, 11, 3, 11, 17, 1]
        # First row is the explanatory variable, second is noise
        X = np.array([[11, 3, 11, 3, 11, 3, 0, 3],
                      [0, 0, 1, 1, 0, 0, 1, 1]])
        X = X.reshape(8, 2)
        bin_mtx = msa_fun.make_bin_mtx(X, AA_TABLE)
        clf = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.99,
                            n_jobs=2, max_iter=100, random_state=42)
        clf.fit(bin_mtx, Y)
        dfs = corrmut.calc_degrees_freedom(clf)
        assert dfs == expected_dfs


class TestBIC():
    """
    Class to test the corrmut.calc_bic function
    """

    def test_bic_one(self):
        # Model with 100 degrees of freedom (default)
        logprobs = np.array([np.log(0.99), np.log(0.90), np.log(0.95),
                             np.log(0.10), np.log(0.05), np.log(0.15)])
        dfs = 100
        n_obs = len(logprobs)
        exp_bic = 193.900229
        bic = corrmut.calc_bic(logprobs, dfs, n_obs)
        assert isclose(exp_bic, bic, rel_tol=1e-6)

    def test_bic_two(self):
        # Same data, but with a model with 0 degrees of freedom
        # (e.g. empty model, only intercepts)
        logprobs = np.array([np.log(0.99), np.log(0.90), np.log(0.95),
                             np.log(0.10), np.log(0.05), np.log(0.15)])
        dfs = 0
        n_obs = len(logprobs)
        exp_bic = 14.724282
        bic = corrmut.calc_bic(logprobs, dfs, n_obs)
        assert isclose(exp_bic, bic, rel_tol=1e-6)


###############################
# Log-likelihood calculations #
###############################


class TestComputeLlhs():
    """
    Class to test the corrmut.compute_llhs function
    """

    ###########
    # Hard EM #
    ###########
    def test_llhs_hard(self):

        alts = np.log(np.array([[0.90, 1e-3], [0.99, 1e-4]]))
        nulls = np.log(np.array([[1e-2, 0.80], [1e-3, 0.85]]))
        labels = np.array([[1, 0], [1, 0]])

        exp_w_alt = np.array([[np.log(0.90), 0],
                              [np.log(0.99), 0]])
        exp_w_null = np.array([[0, np.log(0.80)],
                               [0, np.log(0.85)]])
        w_alt, w_null = corrmut.compute_llhs(labels, alts, nulls)

        assert np.allclose(w_alt, exp_w_alt, rtol=1e-6)
        assert np.allclose(w_null, exp_w_null, rtol=1e-6)

    ###########
    # Soft EM #
    ###########

    def test_llhs_soft(self):
        # Like the previous example, but labels are not rounded
        alts = np.log(np.array([[0.90, 1e-3], [0.99, 1e-4]]))
        nulls = np.log(np.array([[1e-2, 0.80], [1e-3, 0.85]]))
        labels = np.array([[0.90, 1e-3], [0.99, 1e-6]])

        exp_w_alt = np.array([[np.log(0.90) * 0.90, np.log(1e-3) * 1e-3],
                              [np.log(0.99) * 0.99, np.log(1e-4) * 1e-6]])
        exp_w_null = np.array(
            [[np.log(1e-2) * (1 - 0.90), np.log(0.80) * (1 - 1e-3)],
             [np.log(1e-3) * (1 - 0.99), np.log(0.85) * (1 - 1e-6)]])

        w_alt, w_null = corrmut.compute_llhs(labels, alts, nulls)

        assert np.allclose(w_alt, exp_w_alt, rtol=1e-6)
        assert np.allclose(w_null, exp_w_null, rtol=1e-6)
