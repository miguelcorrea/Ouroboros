"""
Module containing functions to digest and validate input
"""
import os
import warnings
import json
import numpy as np


def validate_alignments(num_mtx_a, num_mtx_b):
    """
    Perform sanity checks on input alignments
    """
    assert not np.array_equal(num_mtx_a, num_mtx_b), """Alignments A and B
    are identical"""
    n_obs_a = num_mtx_a.shape[0]
    n_obs_b = num_mtx_b.shape[0]
    assert n_obs_a == n_obs_b, f"""Different number of sequences in MSA A
    ({n_obs_a}) and MSA B ({n_obs_b})"""
    assert n_obs_a > 1, "Only one sequence in the input alignments"


def validate_contact_mtx(msa_a, msa_b, contact_mtx):
    """
    Perform sanity check on input contact matrix
    """

    # Verify that the input contact matrix has the expected dimensions
    assert contact_mtx.shape == (msa_a.shape[1], msa_b.shape[1]), f"""Contact
    matrix is expected to have dimensions
    {msa_a.shape[1]} x {msa_b.shape[1]} (MSA A columns x MSA B columns);
    found {contact_mtx.shape} instead"""
    # Values in the contact matrix must be either 1 or 0
    assert set(np.unique(contact_mtx)) == {0, 1}, """ AAA """


def read_args(json_path):
    """
    Read JSON file containing analysis parameters
    """
    with open(json_path, 'r') as source:
        args = json.load(source)
    return args


def digest_args(args):
    """
    Process and validate input parameters
    """
    io_path = args['io']
    msa_a_path, msa_b_path = digest_msa_paths(args)

    gap_threshold = digest_gap_threshold(args)
    int_frac = digest_int_frac(args)
    init = digest_init(args)
    mode = digest_mode(args)
    test = digest_test(args)
    int_limit = digest_int_limit(args, test)

    contact_mtx = digest_contact_mtx(args)

    n_jobs = digest_n_jobs(args)
    n_starts = digest_n_starts(args, init)
    dfmax = digest_dfmax(args)

    max_init_iters = digest_max_init_iters(args)
    max_reg_iters = digest_max_reg_iters(args)
    predict_contacts = digest_pred_contacts(args)

    return io_path, msa_a_path, msa_b_path, gap_threshold, int_frac, init, \
        mode, test, int_limit, contact_mtx, n_jobs, n_starts, dfmax, max_init_iters,\
        max_reg_iters, predict_contacts


def digest_msa_paths(args):
    if 'msa1' in args.keys():
        msa_a_path = args['msa1']
    else:
        raise KeyError('Missing mandatory parameter: msa1')
    if 'msa2' in args.keys():
        msa_b_path = args['msa2']
    else:
        raise KeyError('Missing mandatory parameter: msa2')

    if not os.path.isfile(msa_a_path):
        raise ValueError('Path to MSA A is not a file')
    if not os.path.isfile(msa_b_path):
        raise ValueError('Path to MSA B is not a file')
    if msa_a_path == msa_b_path:
        raise ValueError('Path to MSA A and MSA B are the same')
    return msa_a_path, msa_b_path


def digest_int_frac(args):
    if 'int_frac' in args.keys():
        int_frac = args['int_frac']
    else:
        raise KeyError('Missing mandatory parameter: int_frac')

    if int_frac <= 0 or int_frac >= 1:
        raise ValueError(f'Invalid value of int_frac {int_frac}')
    return int_frac


def digest_init(args):
    if 'init' in args.keys():
        init = args['init']
    else:
        raise KeyError('Missing mandatory parameter: init')

    if init != "warm" and init != "random":
        raise ValueError(f"""Invalid value of init: {init}.
            Only warm and random accepted""")
    return init


def digest_mode(args):
    if 'mode' in args.keys():
        mode = args['mode']
    else:
        raise KeyError('Missing mandatory parameter: mode')

    if mode != "hard" and mode != "soft":
        raise ValueError(f"""Invalid value of mode: {mode}.
            Only hard and soft accepted""")
    if mode == "hard":
        warnings.warn("""Chosen hard EM. This option is at the moment less
            extensively tested than soft EM.""", UserWarning)
    return mode


def digest_test(args):
    if 'test' in args.keys():
        if type(args['test']) == bool:
            test = args['test']
        else:
            raise ValueError(f"Invalid, non-boolean value for test parameter: {args['test']}")
    else:
        test = False
    return test


def digest_gap_threshold(args, default=0.5):
    if 'gap_threshold' in args.keys():
        gap_threshold = args['gap_threshold']
        if gap_threshold >= 1 or gap_threshold <= 0:
            raise ValueError(f"""Gap threshold value {gap_threshold} outside
                bounds""")
    else:
        gap_threshold = default
    return gap_threshold


def digest_int_limit(args, test):
    if 'int_limit' in args.keys() and test is True:
        int_limit = args['int_limit']
        if int_limit < 0:
            raise ValueError(f'Invalid value of int_limit: {int_limit}')
    elif test is True:
        raise ValueError(
            'int_limit argument is mandatory when using test mode')
    elif 'int_limit' in args.keys():
        warnings.warn(
            'Passed value for int_limit without setting test to True; ignoring option')
        int_limit = None
    else:
        int_limit = None

    return int_limit


def digest_contact_mtx(args):
    if 'contact_mtx' in args.keys():
        if 'predict_contacts' in args.keys() and args['predict_contacts']:
            contact_mtx = args['contact_mtx']
            if not os.path.isfile(contact_mtx):
                raise ValueError('Path to contact matrix is not a file')
        else:
            warnings.warn("""Given a contact matrix, but not asked to predict
                contacts; ignoring contact matrix""", UserWarning)
            contact_mtx = None
    else:
        contact_mtx = None
    return contact_mtx


def digest_n_starts(args, init, default=5):
    if 'n_starts' in args.keys():
        if init == 'warm':
            n_starts = None
        elif init == 'random':
            n_starts = args['n_starts']
            if n_starts < 1:
                raise ValueError(f'Invalid value of n_starts: {n_starts}')
    elif init == 'random':
        n_starts = default
    else:  # Not applicable in warm start
        n_starts = None
    return n_starts


def digest_n_jobs(args, default=2):
    if 'n_jobs' in args.keys():
        n_jobs = args['n_jobs']
        if n_jobs < 1:
            raise ValueError(f'n_jobs value {n_jobs} outside bounds')
    else:
        n_jobs = default
    cpus = os.cpu_count()
    if n_jobs > cpus:
        warnings.warn(f"""n_jobs value {n_jobs} greater than available number
            of CPUs ({cpus}); setting value to {cpus}""")
        n_jobs = cpus
    return n_jobs


def digest_max_init_iters(args, default=100, recommend_min=50):
    if 'max_init_iters' in args.keys():
        max_init_iters = args['max_init_iters']
        if max_init_iters <= 1:
            raise ValueError(f"""Invalid value of max_init_iters:
                {max_init_iters}""")
        if max_init_iters < recommend_min:
            warnings.warn(f"""Low value ({max_init_iters}) of max_init_iters:
                may lead to improper model selection""")
    else:
        max_init_iters = default
    return max_init_iters


def digest_max_reg_iters(args, default=1000, recommend_min=750):
    if 'max_reg_iters' in args.keys():
        max_reg_iters = args['max_reg_iters']
        if max_reg_iters <= 1:
            raise ValueError(f"""Invalid value of max_reg_iters:
                {max_reg_iters}""")
        if max_reg_iters <= recommend_min:
            warnings.warn(f"""Low value ({max_reg_iters}) of max_reg_iters:
             may lead to optimization problems""")
    else:
        max_reg_iters = default
    return max_reg_iters


def digest_pred_contacts(args, default=False):
    if 'predict_contacts' in args.keys():
        if type(args['predict_contacts']) == bool:
            predict_contacts = args['predict_contacts']
        else:
            raise ValueError(f"""Invalid, non-boolean value for
                predict_contact parameter: {args['predict_contacts']}""")
    else:
        predict_contacts = default
    return predict_contacts

########################
# EM keyword arguments #
########################


def pack_em_kwargs(args, true_labels):
    """
    Prepare keyword arguments for EM loop
    """
    em_kwargs = {}

    em_kwargs['tol'] = digest_tol(args)
    em_kwargs['max_iters'] = digest_max_iters(args)
    em_kwargs['dfmax'] = digest_dfmax(args)
    em_kwargs['true_labels'] = true_labels

    return em_kwargs


def digest_tol(args, default=5e-3):
    if 'tol' in args.keys():
        tol = args['tol']
        if tol <= 0:
            raise ValueError(f"Invalid tolerance value: {tol}")
        machine_epsilon = np.finfo(float).eps
        if tol <= machine_epsilon:
            raise ValueError(f"Tolerance set below machine precision")
    else:
        tol = default
    return tol


def digest_max_iters(args, default=20):
    if 'max_iters' in args.keys():
        max_iters = args['max_iters']
        if max_iters <= 1:
            raise ValueError(f"Invalid max_iters value: {max_iters}")
    else:
        max_iters = default
    return max_iters


def digest_dfmax(args, default=100):
    if 'dfmax' in args.keys():
        dfmax = args['dfmax']
        if dfmax <= 1:
            raise ValueError(f'Invalid value of dfmax: {dfmax}')
        if dfmax >= 120:
            warnings.warn(f"""High setting of dfmax ({dfmax}).
                Allowing many degrees of freedom might lead to selection of
                overtly complex models.""")
    else:
        dfmax = default
    return dfmax
