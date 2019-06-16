#!/usr/bin/python
"""
Preprocessing of multiple sequence alignments (MSAs) for correlated
mutations analysis.

@author: Miguel Correa
"""
import os

import numpy as np

import msa_fun
import globalvars

def process(aln, gap_threshold, aa_table):
    """
    Auxiliary function for main()
    """
    aln, gappy_idxs = msa_fun.del_gappy_cols(aln, gap_threshold=gap_threshold)

    num_mtx = msa_fun.make_num_mtx(aln, aa_table)
    num_mtx, constant_idxs = msa_fun.del_constant_cols(num_mtx)
    bin_mtx = msa_fun.make_bin_mtx(num_mtx, aa_table)

    return num_mtx, bin_mtx, gappy_idxs, constant_idxs


def process_contact_mtx(contact_mtx, gappy_idxs_a, constant_idxs_a,
                        gappy_idxs_b, constant_idxs_b):
    """
    Function to remove positions from the contact matrix that have been removed
    from the alignments
    """
    contact_mtx = np.delete(contact_mtx, gappy_idxs_a, axis=0)
    contact_mtx = np.delete(contact_mtx, constant_idxs_a, axis=0)

    contact_mtx = np.delete(contact_mtx, gappy_idxs_b, axis=1)
    contact_mtx = np.delete(contact_mtx, constant_idxs_b, axis=1)

    return contact_mtx


def main(msa_a, msa_b, results_dir, gap_threshold=0.5, contact_mtx=None,
         aa_table=globalvars.AA_TABLE):
    """
    Convenience function to preprocess multiple sequence alignments
    and write them to disk.

    Arguments
    ---------
    msa_a:         TabularMSA object
    msa_b:         TabularMSA object
    gap_threshold: float. Gap frequency threshold: columns that have a gap
                       frequency equal or greater than this value will be
                       removed. By default, 0.5
    results_dir:   str, path to where files will be stored
    aa_table:      aa_table: dictionary
                       Contains the mapping to convert amino acids into numeric
                       factors

    Returns
    ---------
    num_mtx_a: array-like, MSA in numeric matrix form
    bin_mtx_a: array-like, MSA in binary matrix form
    num_mtx_b: array-like, MSA in numeric matrix form
    bin_mtx_b: array-like, MSA in binary matrix form
    """
    num_mtx_a, bin_mtx_a, gappy_idxs_a, constant_idxs_a = process(
        msa_a, gap_threshold, aa_table)

    num_mtx_b, bin_mtx_b, gappy_idxs_b, constant_idxs_b = process(
        msa_b, gap_threshold, aa_table)

    num_a_path = os.path.join(results_dir, "num_mtx_a.csv")
    np.savetxt(num_a_path, num_mtx_a, delimiter=",")
    bin_a_path = os.path.join(results_dir, "bin_mtx_a.csv")
    np.savetxt(bin_a_path, bin_mtx_a, delimiter=",")

    num_b_path = os.path.join(results_dir, "num_mtx_b.csv")
    np.savetxt(num_b_path, num_mtx_b, delimiter=",")
    bin_b_path = os.path.join(results_dir, "bin_mtx_b.csv")
    np.savetxt(bin_b_path, bin_mtx_b, delimiter=",")

    if contact_mtx is not None:
        processed_contact_mtx = process_contact_mtx(contact_mtx, gappy_idxs_a,
                                                    constant_idxs_a,
                                                    gappy_idxs_b,
                                                    constant_idxs_b)
        processed_mtx_path = os.path.join(
            results_dir, "processed_contact_mtx.csv")
        np.savetxt(processed_mtx_path,
                   processed_contact_mtx, delimiter=',')
        return num_mtx_a, bin_mtx_a, num_mtx_b, bin_mtx_b, processed_contact_mtx
    else:
        return num_mtx_a, bin_mtx_a, num_mtx_b, bin_mtx_b
