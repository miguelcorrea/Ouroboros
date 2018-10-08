"""
Unit tests for the input_handling module.
"""

import numpy as np
import inspect
import pytest

import os
import sys

# PARENT = [os.path.join('..')]
# sys.path.extend(PARENT)
import input_handling
from skbio import TabularMSA, Protein

class TestAlignmentValidation():

    def test_ok_alns(self):
        # Check that it does not fail on valid input
        aln_a = np.array([[0, 1, 2, 3, 4], [0, 1, 3, 4, 5],
                          [0, 1, 3, 3, 4], [1, 2, 3, 1, 2]])
        aln_b = np.array([[5, 4, 3, 2, 1], [6, 3, 4, 2, 1],
                          [7, 4, 5, 6, 1], [5, 6, 8, 9, 2]])
        input_handling.validate_alignments(aln_a, aln_b)

    def test_identical_alns(self):
        # Check that it fails if given two identical alignments
        aln_a = np.array([[0, 1, 2, 3], [1, 0, 2, 4],
                          [0, 1, 3, 5]])
        aln_b = aln_a
        with pytest.raises(AssertionError):
            input_handling.validate_alignments(aln_a, aln_b)

    def test_diff_len_aln(self):
        # Check that it fails if given two alignments of different lengths
        aln_a = np.array([[0, 1, 2, 3], [1, 0, 2, 4],
                          [0, 1, 3, 5]])
        aln_b = np.array([[0, 1, 2, 4], [1, 3, 1, 4],
                          [1, 1, 2, 3], [3, 4, 5, 6]])
        with pytest.raises(AssertionError):
            input_handling.validate_alignments(aln_a, aln_b)

    def test_one_seq(self):
        # Check that it fails if given only one sequence
        aln_a = np.array([[0, 1, 2, 3, 4, 5]])
        aln_b = np.array([[4, 5, 3, 2, 1]])
        with pytest.raises(AssertionError):
            input_handling.validate_alignments(aln_a, aln_b)


class TestContactMtxValidation():

    def test_ok(self):
        msa_a = TabularMSA([Protein('ALM'), Protein('AML'), Protein('LAM')])
        msa_b = TabularMSA([Protein('MRT'), Protein('MTR'), Protein('TRM')])
        contact_mtx = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        input_handling.validate_contact_mtx(msa_a, msa_b, contact_mtx)

    def test_mismatch(self):
        msa_a = TabularMSA([Protein('ALMM'), Protein('AMLM'), Protein('LAML')])
        msa_b = TabularMSA([Protein('MRT'), Protein('MTR'), Protein('TRM')])
        contact_mtx = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        with pytest.raises(AssertionError):
            input_handling.validate_contact_mtx(msa_a, msa_b, contact_mtx)

    def test_nonbinary(self):
        msa_a = TabularMSA([Protein('ALM'), Protein('AML'), Protein('LAM')])
        msa_b = TabularMSA([Protein('MRT'), Protein('MTR'), Protein('TRM')])
        contact_mtx = np.array([[1, 0, 0], [10, 0, 0], [0, 0, 0]])
        with pytest.raises(AssertionError):
            input_handling.validate_contact_mtx(msa_a, msa_b, contact_mtx)

####################################################
# Tests for functions checking and digesting input #
####################################################


class TestDigestIntFrac():

    def test_ok(self):
        # Check that it works over a range of valid values
        for i in [1e-6, 0.1, 0.25, 0.5, 0.75, 0.99]:
            args = {'int_frac': i}
            f_int = input_handling.digest_int_frac(args)
            assert f_int == i

    def test_wrong(self):
        # Check that it fails with invalid values
        for i in [-1, 0, 1, 2]:
            args = {'int_frac': i}
            with pytest.raises(ValueError):
                _ = input_handling.digest_int_frac(args)

    def test_missing(self):
        # Check that it fails int_frac is not given
        # (it is a mandatory argument)
        args = {'in_frac': 0.99}
        with pytest.raises(KeyError):
            _ = input_handling.digest_int_frac(args)


class TestDigestInit():

    def test_ok(self):
        # Check that it works with valid values
        for i in ["warm", "random"]:
            args = {'init': i}
            init = input_handling.digest_init(args)
            assert init == i

    def test_wrong(self):
        # Check that it fails with invalid value
        for i in ["wamr", "rando", 1]:
            args = {'init': i}
            with pytest.raises(ValueError):
                _ = input_handling.digest_init(args)

    def test_missing(self):
        # Check that it fails if init is not given
        # (it is a mandatory argument)
        args = {'ini': 'warm'}
        with pytest.raises(KeyError):
            _ = input_handling.digest_init(args)


class TestDigestMode():

    def test_ok(self):
        # Check that it work with valid values
        for i in ["soft", "hard"]:
            args = {"mode": i}
            mode = input_handling.digest_mode(args)
            assert mode == i

    def test_wrong(self):
        # Check that it fails on invalid values
        for i in ["sotf", "hars", 1]:
            args = {'mode': i}
            with pytest.raises(ValueError):
                _ = input_handling.digest_mode(args)

    def test_missing(self):
        # Check that it fails if mode is missing
        # (it is an invalid value)
        args = {'mod': 'soft'}
        with pytest.raises(KeyError):
            _ = input_handling.digest_mode(args)


class TestDigestTest():

    def test_true(self):
        args = {'test': True}
        test = input_handling.digest_test(args)
        assert test is True

    def test_false(self):
        args = {'test': False}
        test = input_handling.digest_test(args)
        assert test is False

    def test_wrong_one(self):
        args = {'test': 'false'}
        with pytest.raises(ValueError):
            _ = input_handling.digest_test(args)

    def test_wrong_two(self):
        args = {'test': 1}
        with pytest.raises(ValueError):
            _ = input_handling.digest_test(args)

    def test_missing(self):
        # Check that it returns False is test is missing
        args = {'init': 'warm'}
        test = input_handling.digest_test(args)
        assert test is False


class TestDigestGapThreshold():

    def test_ok(self):
        # Check that it works on valid input
        for i in [1e-6, 0.1, 0.25, 0.5, 0.75, 0.99]:
            args = {'gap_threshold': i}
            gap_thr = input_handling.digest_gap_threshold(args)
            assert gap_thr == i

    def test_wrong(self):
        # Check that it fails on invalid input
        for i in [-1, 0, 1, 2]:
            args = {'gap_threshold': i}
            with pytest.raises(ValueError):
                _ = input_handling.digest_gap_threshold(args)

    def test_missing(self):
        # Check that it returns the default value if gap_threshold is missing
        args = {'init': 'warm', 'mode': 'soft'}
        sig = inspect.signature(input_handling.digest_gap_threshold)
        gap_thr = input_handling.digest_gap_threshold(args)
        assert gap_thr == sig.parameters['default'].default


class TestDigestIntLimit():

    def test_ok_one(self):
        # Check that it returns correct value
        args = {'int_limit': 100}
        int_limit = input_handling.digest_int_limit(args, True)
        assert int_limit == 100

    def test_wrong(self):
        # Check that it fails on invalid value
        args = {'int_limit': -1}
        with pytest.raises(ValueError):
            _ = input_handling.digest_int_limit(args, True)

    def test_missing_1(self):
        # Check that it fails if test is True but int_limit is not provided
        args = {'mode': 'soft'}
        with pytest.raises(ValueError):
            _ = input_handling.digest_int_limit(args, True)

    def test_missing_2(self):
        # Check that it returns None if test is False but int_limit is provided
        # anyway
        args = {'int_limit': 100}
        int_limit = input_handling.digest_int_limit(args, False)
        assert int_limit is None

    def test_missing_3(self):
        # Check that it returns None if test is False and int_limit is
        # not provided
        args = {'mode': 'soft'}
        int_limit = input_handling.digest_int_limit(args, False)
        assert int_limit is None


class TestDigestContactMtx():

    def test_no_mtx(self):
        args = {'init': 'warm'}
        contact_mtx = input_handling.digest_contact_mtx(args)
        assert contact_mtx is None

    def test_no_pred(self):
        args = {'predict_contacts': False, 'contact_mtx': 'mtx.csv'}
        contact_mtx = input_handling.digest_contact_mtx(args)
        assert contact_mtx is None


class TestDigestNStarts():

    def test_random(self):
        # Test it returns the given value if init == random
        args = {'n_starts': 10}
        n_starts = input_handling.digest_n_starts(args, 'random')
        assert n_starts == 10

    def test_warm(self):
        # Test it returns None if init == warm
        args = {'n_starts': 10}
        n_starts = input_handling.digest_n_starts(args, 'warm')
        assert n_starts is None

    def test_default(self):
        # Test it returns default value if n_starts is not provided
        args = {'mode': 'soft'}
        n_starts = input_handling.digest_n_starts(args, 'random')
        sig = inspect.signature(input_handling.digest_n_starts)
        assert n_starts == sig.parameters['default'].default

    def test_wrong(self):
        # Test it fails on invalid input
        wrong_starts = [0.5, 0, -1]
        for i in wrong_starts:
            args = {'n_starts': i}
            with pytest.raises(ValueError):
                _ = input_handling.digest_n_starts(args, 'random')


class TestDigestNJobs():

    def test_ok(self):
        args = {'n_jobs': 3}
        n_jobs = input_handling.digest_n_jobs(args)
        assert n_jobs == 3

    def test_wrong(self):
        args = {'n_jobs': 0}
        with pytest.raises(ValueError):
            _ = input_handling.digest_n_jobs(args)

    def test_default(self):
        args = {'mode': 'soft'}
        n_jobs = input_handling.digest_n_jobs(args)
        sig = inspect.signature(input_handling.digest_n_jobs)
        assert n_jobs == sig.parameters['default'].default


class TestDigestMaxInitIters():

    def test_ok(self):
        args = {'max_init_iters': 120}
        max_init_iters = input_handling.digest_max_init_iters(args)
        assert max_init_iters == 120

    def test_wrong(self):
        args = {'max_init_iters': 0}
        with pytest.raises(ValueError):
            _ = input_handling.digest_max_init_iters(args)

    def test_default(self):
        args = {'mode': 'soft'}
        max_init_iters = input_handling.digest_max_init_iters(args)
        sig = inspect.signature(input_handling.digest_max_init_iters)
        assert max_init_iters == sig.parameters['default'].default


class TestDigestMaxRegIters():

    def test_ok(self):
        args = {'max_reg_iters': 1020}
        max_reg_iters = input_handling.digest_max_reg_iters(args)
        assert max_reg_iters == args['max_reg_iters']

    def test_wrong(self):
        args = {'max_reg_iters': 0}
        with pytest.raises(ValueError):
            _ = input_handling.digest_max_reg_iters(args)

    def test_default(self):
        args = {'mode': 'soft'}
        max_init_iters = input_handling.digest_max_reg_iters(args)
        sig = inspect.signature(input_handling.digest_max_reg_iters)
        assert max_init_iters == sig.parameters['default'].default


class TestDigestPredContacts():

    def test_true(self):
        args = {'predict_contacts': True}
        pred_contacts = input_handling.digest_pred_contacts(args)
        assert pred_contacts is True

    def test_false(self):
        args = {'predict_contacts': False}
        pred_contacts = input_handling.digest_pred_contacts(args)
        assert pred_contacts is False

    def test_wrong_one(self):
        args = {'predict_contacts': 'false'}
        with pytest.raises(ValueError):
            _ = input_handling.digest_pred_contacts(args)

    def test_wrong_two(self):
        args = {'predict_contacts': 1}
        with pytest.raises(ValueError):
            _ = input_handling.digest_pred_contacts(args)

    def test_missing(self):
        args = {'init': 'warm'}
        pred_contacts = input_handling.digest_pred_contacts(args)
        assert pred_contacts is False


class TestDigestTol():

    def test_ok(self):
        args = {'tol': 1e-6}
        tol = input_handling.digest_tol(args)
        assert tol == args['tol']

    def test_wrong(self):
        args = {'tol': 0}
        with pytest.raises(ValueError):
            _ = input_handling.digest_tol(args)

    def test_fail_below_epsilon(self):
        args = {'tol': (np.finfo(float).eps / 2)}
        with pytest.raises(ValueError):
            _ = input_handling.digest_tol(args)

    def test_default(self):
        args = {'mode': 'soft'}
        tol = input_handling.digest_tol(args)
        sig = inspect.signature(input_handling.digest_tol)
        assert tol == sig.parameters['default'].default


class TestDigestMaxIters():

    def test_ok(self):
        args = {'max_iters': 10}
        max_iters = input_handling.digest_max_iters(args)
        assert max_iters == args['max_iters']

    def test_wrong(self):
        args = {'max_iters': 1}
        with pytest.raises(ValueError):
            _ = input_handling.digest_max_iters(args)

    def test_default(self):
        args = {'mode': 'soft'}
        max_iters = input_handling.digest_max_iters(args)
        sig = inspect.signature(input_handling.digest_max_iters)
        assert max_iters == sig.parameters['default'].default


class TestDigestDfmax():

    def test_ok(self):
        args = {'dfmax': 50}
        dfmax = input_handling.digest_dfmax(args)
        assert dfmax == args['dfmax']

    def test_wrong(self):
        args = {'dfmax': 0}
        with pytest.raises(ValueError):
            _ = input_handling.digest_dfmax(args)

    def test_default(self):
        args = {'mode': 'soft'}
        dfmax = input_handling.digest_dfmax(args)
        sig = inspect.signature(input_handling.digest_dfmax)
        assert dfmax == sig.parameters['default'].default
