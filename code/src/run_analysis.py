#!/usr/bin/python
"""
Main script for running combined expectation maximization - correlated mutations algorithm.

Usage: python run_analysis.py $PARAMETER_FILE_PATH

@author: Miguel Correa Marrero
"""
# Allows plotting without a running X server
# Prevent skbio from setting the matplotlib backend
import matplotlib as mpl
mpl.use("Agg")

import os
from sys import argv
import json
from random import seed
import warnings

import numpy as np

import output
import globalvars
import preprocess
import corrmut
import contacts

from skbio import TabularMSA, Protein


def read_args(json_path):
    """
    Read JSON file containing analysis parameters
    """
    with open(json_path, 'r') as source:
        args = json.load(source)
    return args


def digest_args(args):
    """
    Process input parameters and set defaults
    """
    io_path = args['io']
    msa_a_path = args['msa1']
    msa_b_path = args['msa2']
    if not os.path.isfile(msa_a_path):
        raise ValueError('Path to MSA A is not a file')
    if not os.path.isfile(msa_b_path):
        raise ValueError('Path to MSA B is not a file')
    if msa_a_path == msa_b_path:
        raise ValueError('Path to MSA A and MSA B are the same')

    int_frac = args['int_frac']
    init = args['init']
    mode = args['mode']
    test = bool(args['test'])

    keys = list(args.keys())

    if 'gap_threshold' in keys:
        gap_threshold = args['gap_threshold']
        if gap_threshold > 0.99 or gap_threshold < 0:
            raise ValueError(f'Gap threshold value {gap_threshold} outside bounds')
    else:
        gap_threshold = 0.5

    if 'int_limit' in keys:
        int_limit = args['int_limit']
        if int_limit < 0:
            raise ValueError(f'Invalid value of int_limit: {int_limit}')
    elif test:
        raise ValueError(
            'int_limit argument is mandatory when using test mode')

    if 'contact_mtx' in keys:
        contact_mtx = args['contact_mtx']
        if not os.path.isfile(contact_mtx):
            raise ValueError('Path to contact matrix is not a file')
    elif test:
        warnings.warn(
            'Running in test mode, but without a ground truth contact matrix')
        contact_mtx = None
    else:
        contact_mtx = None

    if 'prior_ints' in keys:
        prior_ints = args['prior_ints']
        warnings.warn("prior_ints option is not throughly tested: use with caution")
        if not os.path.isfile(prior_ints):
            raise ValueError(
                'Path to file containing prior information is not a file')
    else:
        prior_ints = None

    if 'n_jobs' in keys:
        n_jobs = args['n_jobs']
    else:
        n_jobs = 2
    if 'n_starts' in keys:
        n_starts = args['n_starts']
    elif init == 'random':
        n_starts = 5
        if n_starts < 1:
            raise ValueError(f'Invalid value of n_starts: {n_starts}')
    else:  # Not applicable in warm start
        n_starts = None

    if 'dfmax' in keys:
        dfmax = args['dfmax']
        if dfmax < 0:
            raise ValueError(f'Invalid value of dfmax: {dfmax}')
    else:
        dfmax = 100

    return io_path, msa_a_path, msa_b_path, gap_threshold, int_frac, init, \
        mode, test, int_limit, contact_mtx, n_jobs, n_starts, dfmax, prior_ints


def pack_em_kwargs(args, true_labels):
    """
    Prepare keyword arguments for EM loop
    """
    em_args = {}

    keys = list(args.keys())

    if 'tol' in keys:
        em_args['tol'] = args['tol']
    else:
        em_args['tol'] = 0.005

    if 'max_iters' in keys:
        em_args['max_iters'] = args['max_iters']
    else:
        em_args['max_iters'] = 20

    if 'dfmax' in keys:
        em_args['dfmax'] = args['dfmax']
    else:
        em_args['dfmax'] = 100

    em_args['true_labels'] = true_labels

    return em_args

if __name__ == "__main__":

    print(globalvars.LOGO)

    #########
    # Setup #
    #########
    seed(42)
    np.random.seed(42)

    # Read and process parameters from JSON file
    args = read_args(argv[1])
    io_path, msa_a_path, msa_b_path, gap_threshold, int_frac, init, mode,\
        test, int_limit, contact_mtx, n_jobs, n_starts, dfmax, prior_ints = digest_args(
            args)

    # Create directory tree
    results_dir = os.path.join(io_path)
    os.mkdir(results_dir)
    checks_dir = os.path.join(results_dir, "output")
    os.mkdir(checks_dir)

    #############################
    # Load and preprocess input #
    #############################
    print("Reading input...")
    msa_a = TabularMSA.read(msa_a_path, constructor=Protein)
    msa_b = TabularMSA.read(msa_b_path, constructor=Protein)
    if contact_mtx:
        true_contact_mtx = np.loadtxt(contact_mtx, delimiter=',')
        num_mtx_a, bin_mtx_a, num_mtx_b, bin_mtx_b,\
            true_contact_mtx = preprocess.main(msa_a, msa_b, results_dir,
                                               contact_mtx=true_contact_mtx,
                                               gap_threshold=gap_threshold)
    else:
        num_mtx_a, bin_mtx_a, num_mtx_b, bin_mtx_b = preprocess.main(
            msa_a, msa_b, results_dir, gap_threshold=gap_threshold)

    n_obs = num_mtx_a.shape[0]
    if test:
        if int_limit > 0:
            if int_limit > n_obs:
                raise ValueError(f"""Invalid value of int_limit {int_limit}:
                                 greater than the number of sequences""")
            true_labels = [1 if idx <=
                           int_limit else 0 for idx in range(n_obs)]
        else:  # For test cases where all sequence pairs are non-interacting
            true_labels = [0 for item in range(n_obs)]
    else:
        true_labels = None

    # Read file containing prior information about the interactions
    if prior_ints:
        prior_inters = np.genfromtxt(prior_ints, delimiter=',')
    else:
        prior_inters = None

    em_args = pack_em_kwargs(args, true_labels)

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
                                                                                   dfmax, prior_inters)

        print('Start EM loop...')
        labels_per_iter, alt_llhs_per_iter, \
            null_llhs_per_iter, contacts_per_iter = corrmut.em_loop(num_mtx_a, num_mtx_b,
                                                                    bin_mtx_a, bin_mtx_b,
                                                                    init_labels,
                                                                    int_frac, mode,
                                                                    checks_dir, n_jobs,
                                                                    **em_args,
                                                                    fixed_alphas_a=alphas_a,
                                                                    fixed_alphas_b=alphas_b,
                                                                    prior_inters=prior_inters)

        print('Predicting contacts with final protein-protein interaction predictions...')
        final_couplings,\
            final_contact_mtx = corrmut.contact_prediction(num_mtx_a, bin_mtx_b,
                                                           num_mtx_b, bin_mtx_a,
                                                           labels_per_iter[-1], mode,
                                                           n_jobs, dfmax)
        np.savetxt(os.path.join(checks_dir, ''.join(
            ['final_contact_mtx', '.csv'])), final_contact_mtx, delimiter=',')

        norm_final_contact_mtx = contacts.normalize_contact_mtx(
            final_contact_mtx)
        np.savetxt(os.path.join(checks_dir, ''.join(
            ['norm_final_contact_mtx', '.csv'])), norm_final_contact_mtx,
            delimiter=',')

        # Insert labels and log-likelihoods from the initial step
        labels_per_iter.insert(0, init_labels)
        alt_llhs_per_iter.insert(0, init_alt_llhs)
        null_llhs_per_iter.insert(0, init_null_llhs)
        contacts_per_iter.insert(0, init_contacts)
        # Add final contact predictions
        contacts_per_iter.append(norm_final_contact_mtx)

        # Compute weighted likelihoods for each iteration
        alt_int_per_iter, \
            null_nonint_per_iter = corrmut.compute_llhs(labels_per_iter,
                                                        alt_llhs_per_iter,
                                                        null_llhs_per_iter,
                                                        mode)
        # Compute weighted likelihoods for the true solution (if available)
        # and create output
        if test:
            alt_true_per_iter, \
                null_true_per_iter = corrmut.compute_llhs(
                    [true_labels] * len(labels_per_iter), alt_llhs_per_iter,
                    null_llhs_per_iter, mode)

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
