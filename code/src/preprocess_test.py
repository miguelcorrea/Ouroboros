"""
Unit tests for the preprocess module.
"""
import numpy as np
import os
import sys
# PARENT = [os.path.join('..')]
# sys.path.extend(PARENT)
import preprocess
from globalvars import AA_TABLE
from skbio import TabularMSA, Protein


class TestProcess():
    """
    Class to test the preprocess.process function
    """

    def test_process_1(self):
        aln = TabularMSA([Protein('AL-'), Protein('VL-'), Protein('MLA')])
        gap_thr = 0.5

        exp_num = [[AA_TABLE['A']], [AA_TABLE['V']], [AA_TABLE['M']]]
        exp_bin = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]

        num_mtx, bin_mtx, gappy_idxs, constant_idxs = preprocess.process(
            aln, gap_thr, AA_TABLE)
        assert np.array_equal(exp_num, num_mtx)
        assert np.array_equal(exp_bin, bin_mtx)
        assert gappy_idxs == [2]
        assert constant_idxs == [1]

    def test_process_2(self):
        # Invert columns 1 and 2 with respect to the previous example
        aln = TabularMSA([Protein('A-L'), Protein('V-L'), Protein('MAL')])
        gap_thr = 0.5

        exp_num = [[AA_TABLE['A']], [AA_TABLE['V']], [AA_TABLE['M']]]
        exp_bin = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]

        num_mtx, bin_mtx, gappy_idxs, constant_idxs = preprocess.process(
            aln, gap_thr, AA_TABLE)
        assert np.array_equal(exp_num, num_mtx)
        assert np.array_equal(exp_bin, bin_mtx)
        assert gappy_idxs == [1]
        assert constant_idxs == [1]


class TestProcessContactMtx():
    """
    Class to test the preprocess.process_contact_mtx
    """

    def test_one(self):

        # All will be removed, except the first column in both alignments
        msa_a = TabularMSA([Protein('DL-'), Protein('KL-'), Protein('DL-')])
        msa_b = TabularMSA([Protein('KT-'), Protein('DT-'), Protein('KT-')])
        contact_mtx = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        exp_contact_mtx = np.array([[1]])
        gap_threshold = 0.5
        num_mtx_a, bin_mtx_a, gappy_idxs_a, constant_idxs_a = preprocess.process(
            msa_a, gap_threshold, AA_TABLE)
        num_mtx_b, bin_mtx_b, gappy_idxs_b, constant_idxs_b = preprocess.process(
            msa_b, gap_threshold, AA_TABLE)
        proc_contact_mtx = preprocess.process_contact_mtx(
            contact_mtx, gappy_idxs_a, constant_idxs_a, gappy_idxs_b, constant_idxs_b)
        print(proc_contact_mtx)
        print(exp_contact_mtx)
        assert np.array_equal(proc_contact_mtx, exp_contact_mtx)
