"""
Unit tests for the msa_fun module
"""
import pytest
import numpy as np
from skbio import TabularMSA, Protein

import os
import sys
PARENT = [os.path.join('..')]
sys.path.extend(PARENT)
import msa_fun
from globalvars import AA_TABLE


class TestMakeNumMtx():

    def test_ok_1(self):
        aln = TabularMSA([Protein('ARN'), Protein('DCE'), Protein('QGH'),
                          Protein('ILK'), Protein('MFP'), Protein('STW'),
                          Protein('YV-')])
        expected_mtx = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                                 [12, 13, 14], [15, 16, 17], [18, 19, 20]])
        num_mtx = msa_fun.make_num_mtx(aln, AA_TABLE)
        assert np.array_equal(num_mtx, expected_mtx) is True

    def test_ok_2(self):
        aln = TabularMSA([Protein('A-'), Protein('V-'), Protein('-V')])
        expected_mtx = np.array([[0.,  20.],
                                 [19.,  20.],
                                 [20.,  19.]])
        num_mtx = msa_fun.make_num_mtx(aln, AA_TABLE)
        assert np.array_equal(num_mtx, expected_mtx) is True


class TestMakeBinMtx():

    def test_ok(self):
        num_mtx = np.array([[0.,  20.],
                            [19.,  20.],
                            [20.,  19.]])
        expected_mtx = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        bin_mtx = msa_fun.make_bin_mtx(num_mtx, AA_TABLE)
        assert np.array_equal(bin_mtx, expected_mtx) is True


class TestDelGappyCols():

    def test_no_gappy_1(self):
        # Output should be identical to input
        aln = TabularMSA([Protein('ELV'), Protein('AVL'),
                          Protein('ALR'), Protein('ELR')])
        out_aln, gappy_idxs = msa_fun.del_gappy_cols(aln, gap_threshold=0.5)
        assert out_aln == aln
        assert len(gappy_idxs) == 0

    def test_no_gappy_2(self):
        # Output should be identical to input
        aln = TabularMSA([Protein('-LV'), Protein('A-L'),
                          Protein('AL-'), Protein('ELR')])
        out_aln, gappy_idxs = msa_fun.del_gappy_cols(aln, gap_threshold=0.5)
        assert out_aln == aln
        assert len(gappy_idxs) == 0

    def test_one_gappy(self):
        aln = TabularMSA([Protein('EL-'), Protein('AV-'),
                          Protein('ALR'), Protein('ELR')])
        exp_aln = TabularMSA([Protein('EL'), Protein('AV'),
                              Protein('AL'), Protein('EL')])
        out_aln, gappy_idxs = msa_fun.del_gappy_cols(aln, gap_threshold=0.5)
        assert exp_aln == out_aln
        assert gappy_idxs == [2]

    def test_all_gappy(self):
        aln = TabularMSA([Protein('---'), Protein('---'),
                          Protein('ALR'), Protein('ELR')])
        with pytest.raises(Exception):
            _ = msa_fun.del_gappy_cols(aln, gap_threshold=0.5)


class TestDelConstantCols():

    def test_no_constant(self):
        # Output should be identical to input
        num_mtx = np.array([[1, 2, 3, 4], [1, 2, 4, 5],
                            [1, 3, 5, 6], [2, 3, 7, 8]])
        out_aln, constant_idxs = msa_fun.del_constant_cols(num_mtx)
        assert np.array_equal(num_mtx, out_aln)
        assert len(constant_idxs) == 0

    def test_one_constant(self):
        num_mtx = np.array([[1, 2, 3, 4], [1, 2, 4, 5],
                            [1, 3, 5, 6], [1, 3, 7, 8]])
        exp_aln = np.array([[2, 3, 4], [2, 4, 5],
                            [3, 5, 6], [3, 7, 8]])
        out_aln, constant_idxs = msa_fun.del_constant_cols(num_mtx)
        assert np.array_equal(exp_aln, out_aln)
        assert constant_idxs == [0]

    def test_all_constant(self):
        num_mtx = np.array([[1, 2, 3, 4], [1, 2, 3, 4],
                            [1, 2, 3, 4], [1, 2, 3, 4]])
        with pytest.raises(Exception):
            _ = msa_fun.del_constant_cols(num_mtx)
