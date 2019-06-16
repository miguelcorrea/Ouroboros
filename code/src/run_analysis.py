#!/usr/bin/python
"""
Main script for running combined expectation maximization - correlated mutations
algorithm.

Usage: python run_analysis.py $PARAMETER_FILE_PATH

__author__ = "Miguel Correa Marrero"
__credits__ = ["Miguel Correa Marrero","Richard G.H Immink","Dick de Ridder",
              "Aalt-Jan van Dijk"]
__maintainer__ = "Miguel Correa Marrero"
__license__ = "BSD-3"
"""

# Allows plotting without a running X server
# Prevent skbio from setting the matplotlib backend
import matplotlib as mpl
mpl.use("Agg")

import os
from sys import argv
from random import seed
import warnings

import numpy as np

import input_handling
import output
import globalvars
import preprocess
import corrmut
import contacts

from skbio import TabularMSA, Protein


def generate_true_labels(int_limit, n_obs):
    """
    Function to generate ground truth labels as specified by int_limit.
    """
    if int_limit > 0:
        if int_limit > n_obs:
            raise ValueError(f"""Invalid value of int_limit {int_limit}:
                             greater than the number of sequences""")
        else:
            true_labels = [1 if idx <=
                           int_limit else 0 for idx in range(n_obs)]
    else:  # Allows test cases where all sequence pairs are non-interacting
        true_labels = [0 for item in range(n_obs)]
    return true_labels


if __name__ == "__main__":

    print(globalvars.LOGO)

    #########
    # Setup #
    #########
    seed(42)
    np.random.seed(42)

    # Read and process parameters from JSON file
    # TODO: allow use of max_init_iters and max_reg_iters parameters!
    args = input_handling.read_args(argv[1])
    io_path, msa_a_path, msa_b_path, gap_threshold, int_frac, init, mode, \
        test, int_limit, contact_mtx, n_jobs, n_starts, dfmax, max_init_iters, \
        max_reg_iters, predict_contacts = input_handling.digest_args(args)

    # Create directory tree
    results_dir = os.path.join(io_path)
    os.mkdir(results_dir)
    checks_dir = os.path.join(results_dir, "output")
    os.mkdir(checks_dir)

    #######################################
    # Load, preprocess and validate input #
    #######################################

    print("Reading and processing input...")
    msa_a = TabularMSA.read(msa_a_path, constructor=Protein)
    msa_b = TabularMSA.read(msa_b_path, constructor=Protein)
    if contact_mtx:
        true_contact_mtx = np.loadtxt(contact_mtx, delimiter=',')
        num_mtx_a, bin_mtx_a, num_mtx_b, bin_mtx_b,\
            true_contact_mtx = preprocess.main(msa_a, msa_b, results_dir,
                                               contact_mtx=true_contact_mtx,
                                               gap_threshold=gap_threshold)
        input_handling.validate_contact_mtx(msa_a, msa_b, contact_mtx)
    else:
        num_mtx_a, bin_mtx_a, num_mtx_b, bin_mtx_b = preprocess.main(
            msa_a, msa_b, results_dir, gap_threshold=gap_threshold)
    input_handling.validate_alignments(num_mtx_a, num_mtx_b)

    if test:
        true_labels = generate_true_labels(int_limit, num_mtx_a.shape[0])
    else:
        true_labels = None

    em_args = input_handling.pack_em_kwargs(args, true_labels)

    ###########################################################
    # Combined expectation-maximization-correlated mutations  #
    ###########################################################

    if init == 'warm':
        print("Initialize model...")
        init_labels, init_alt_llhs,\
            init_null_llhs, init_contacts, alphas_a, alphas_b = corrmut.init_model(num_mtx_a, bin_mtx_b,
                                                                                   num_mtx_b, bin_mtx_a,
                                                                                   mode, init, int_frac,
                                                                                   checks_dir, n_jobs,
                                                                                   dfmax)

        print('Start EM loop...')
        labels_per_iter, alt_llhs_per_iter, \
            null_llhs_per_iter, contacts_per_iter = corrmut.em_loop(num_mtx_a, num_mtx_b,
                                                                    bin_mtx_a, bin_mtx_b,
                                                                    init_labels,
                                                                    int_frac, mode,
                                                                    checks_dir, n_jobs,
                                                                    **em_args,
                                                                    fixed_alphas_a=alphas_a,
                                                                    fixed_alphas_b=alphas_b)

        if predict_contacts:
            print(
                'Predicting contacts with final protein-protein interaction predictions...')
            final_couplings,\
                final_contact_mtx = corrmut.contact_prediction(num_mtx_a, bin_mtx_b,
                                                               num_mtx_b, bin_mtx_a,
                                                               labels_per_iter[
                                                                   -1], mode,
                                                               n_jobs, dfmax)
            np.savetxt(os.path.join(checks_dir, ''.join(
                ['final_contact_mtx', '.csv'])), final_contact_mtx, delimiter=',')

            norm_final_contact_mtx = contacts.normalize_contact_mtx(
                final_contact_mtx)
            np.savetxt(os.path.join(checks_dir, ''.join(
                ['norm_final_contact_mtx', '.csv'])), norm_final_contact_mtx,
                delimiter=',')
            # Add final contact predictions
            contacts_per_iter.append(norm_final_contact_mtx)

        # Insert labels and log-likelihoods from the initial step
        labels_per_iter.insert(0, init_labels)
        alt_llhs_per_iter.insert(0, init_alt_llhs)
        null_llhs_per_iter.insert(0, init_null_llhs)
        contacts_per_iter.insert(0, init_contacts)

        # Compute weighted likelihoods and create output
        alt_int_per_iter, null_nonint_per_iter = corrmut.compute_llhs(labels_per_iter,
                                                                      alt_llhs_per_iter,
                                                                      null_llhs_per_iter)
        if test:
            # Use information about the true solution
            alt_true_per_iter, \
                null_true_per_iter = corrmut.compute_llhs(
                    [true_labels] * len(labels_per_iter), alt_llhs_per_iter,
                    null_llhs_per_iter)

            output.create_output(labels_per_iter, alt_llhs_per_iter,
                                 null_llhs_per_iter, alt_int_per_iter,
                                 null_nonint_per_iter, mode, results_dir, test,
                                 true_labels, alt_true_per_iter,
                                 null_true_per_iter)
        else:
            output.create_output(labels_per_iter, alt_llhs_per_iter,
                                 null_llhs_per_iter, alt_int_per_iter,
                                 null_nonint_per_iter, mode, results_dir, test)

    elif init == 'random':
        if test:
            corrmut.em_wrapper(num_mtx_a, num_mtx_b, bin_mtx_a, bin_mtx_b,
                               n_starts, int_frac, mode,
                               results_dir, n_jobs, dfmax, test, em_args, true_labels)
        else:
            corrmut.em_wrapper(num_mtx_a, num_mtx_b, bin_mtx_a, bin_mtx_b,
                               n_starts, int_frac, mode,
                               results_dir, n_jobs, dfmax, test, em_args)

    print(globalvars.END)
