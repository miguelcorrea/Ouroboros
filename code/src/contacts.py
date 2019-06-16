"""
Module containing functions for intermolecular contact prediction.

@author: Miguel Correa
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef
import warnings

######################
# Contact prediction #
######################


def get_interacting(num_mtx_a, bin_mtx_b, num_mtx_b, bin_mtx_a, labels,
                    mode, soft_threshold=0.5):
    """
    Auxiliary function to select pairs of putatively interacting proteins prior
    to fitting the models.

    Arguments
    ---------
    num_mtx_a, num_mtx_b: array-like. Contains a multiple sequence alignment as
                          a numeric matrix with all proteins under
                          consideration, interacting and non-interacting.
    bin_mtx_a, bin_mtx_b: array-like. Contains a multiple sequence alignment as
                          a binary matrix with all proteins under
                          consideration, interacting and non-interacting.
    labels:               array-like, values of the hidden variables
    mode:                 str, whether to perform soft or hard
                          expectation-maximization
    soft_threshold:       float, hidden variable threshold above which
                          observations are included in the logistic models

    Returns
    -------
    int_num_a, int_num_b: array-like, selected sequence pairs of the MSAs
    int_bin_a, int_bin_b: array-like, selected rows of the predictor matrices
    weights:              array-like, selected values of the hidden variables

    """
    labels = np.asarray(labels)
    if mode == 'hard':
        idxs = np.where(np.asarray(labels) == 1)[0]

    elif mode == 'soft':
        idxs = np.where(labels >= soft_threshold)[0]

    int_num_a = np.take(num_mtx_a, idxs, axis=0)
    int_num_b = np.take(num_mtx_b, idxs, axis=0)

    int_bin_a = np.take(bin_mtx_a, idxs, axis=0)
    int_bin_b = np.take(bin_mtx_b, idxs, axis=0)

    weights = np.take(labels, idxs)

    return int_num_a, int_bin_b, int_num_b, int_bin_a, weights


def select_coefs(model_coefs, idx):
    """
    Auxiliary function for compute_couplings().
    Used to select the desired subvector/submatrix from the vector/matrix
    of model coefficients.
    """
    if len(model_coefs.shape) == 1:  # Binomial case
        sel_coefs = model_coefs[range(idx, idx + 20)]
    else:  # Multinomial case
        sel_coefs = model_coefs[:, range(idx, idx + 20)]
    return sel_coefs, idx + 20


def compute_couplings(models_a, models_b):
    """
    Given logistic models for two multiple sequence alignments, calculate all
    intermolecular coupling strengths between residues.
    The coupling strength between positions i and j is calculated as the 2-norm
    of the concatenation of the coefficient submatrices that describe the
    relationships between the two positions.

    ----------------------------------------------------------------------------
    Reference:
    Ovchinnikov, Sergey, Hetunandan Kamisetty, and David Baker.
    "Robust and accurate prediction of residue–residue interactions across
    protein interfaces using evolutionary information." Elife 3 (2014): e02030
    ----------------------------------------------------------------------------

    Arguments
    ---------
    models_a: list of SGDClassifier objects, one for each analyzed column in
              MSA A
    models_b: list of SGDClassifier objects, one for each analyzed column in
              MSA B

    Returns
    -------
    couplings:   dict, contains intermolecular coupling strengths in the format
                 {"Ai:Bj":float,...}
    contact_mtx: array, 2D matrix of dimensions (models_a, models_b); contains
                 the value of the coupling strength for each pair of positions

    """
    # Dictionary to store couplings between residues
    couplings = {}
    # To keep track of the submatrix we need to take from the matrix of
    # coefficients from protein B

    # Iterate over models / columns of MSA A
    # Variable to keep track of the submatrix we need to take from the matrix
    # of coefficients of models of B
    offset_a = 0
    contact_mtx = np.zeros((len(models_a), len(models_b)))
    for i, model_a in enumerate(models_a):
        # Variable to keep track of the submatrix we need to take from the
        # matrix of coefficients from protein A
        end_point_a = 0

        for j, model_b in enumerate(models_b):
            # Select the relevant submatrices of coefficients, this is,
            # the columns in A that indicate coupling to B and vice versa
            # Taking the 2-norm of a vector and a matrix is equivalent. In case
            # of mismatching dimensions, flatten the matrices into vectors and
            # concatenate them
            sel_coefs_a, end_point_a = select_coefs(model_a.coef_, end_point_a)
            sel_coefs_a = sel_coefs_a.flatten()
            sel_coefs_b, _ = select_coefs(model_b.coef_, offset_a)
            sel_coefs_b = sel_coefs_b.flatten()
            coef_vector = np.concatenate((sel_coefs_a, sel_coefs_b))

            # Calculate coupling strength (as the 2-norm of the vector of
            # coefficients) and store the value in the output
            coupling = np.linalg.norm(coef_vector)
            coupling_name = ''.join(['A', str(i), ':', 'B', str(j)])
            couplings[coupling_name] = coupling
            contact_mtx[i][j] = coupling
        offset_a += 20

    return couplings, contact_mtx


def normalize_contact_mtx(contact_mtx):
    """
    Apply Average Product Correction to the contact matrix.

    See:
    Dunn, Stanley D., Lindi M. Wahl, and Gregory B. Gloor.
    "Mutual information without the influence of phylogeny or entropy
    dramatically improves residue contact prediction."
    Bioinformatics 24.3 (2007): 333-340
    """
    norm_mtx = np.zeros_like(contact_mtx)

    # Precompute means
    mean_coupling = contact_mtx.mean()
    row_means = contact_mtx.mean(axis=1)
    col_means = contact_mtx.mean(axis=0)

    # Iterate over array to compute the corresponding correction
    for idxs, coupling in np.ndenumerate(contact_mtx):
        apc = (row_means[idxs[0]] * col_means[idxs[1]]) / mean_coupling
        corr_coupling = coupling - apc
        norm_mtx[idxs] = corr_coupling
    return norm_mtx


def eval_contact_metrics(true_contact_mtx, pred_contact_mtx, limit=100):
    """
    Calculate contact prediction accuracy for a number X of predicted contacts,
    starting from the strongest ones.
    """

    # Obtain indexes of true contacts
    true_idxs = list(zip(*np.where(true_contact_mtx == 1)))
    if limit > pred_contact_mtx.size:
        warnings.warn("""Limit of predicted contacts greater than contact
             matrix size; truncating""", RuntimeWarning)
        limit = pred_contact_mtx.size

    tpr_per_rank = []
    ppv_per_rank = []
    for i in range(1, limit + 1):
        # How many positive samples do we consider?
        if i > len(true_idxs):  # We have exhausted the positives
            no_positives = len(true_idxs)
        else:
            no_positives = i

        no_true_positives = 0
        no_false_positives = 0
        # Obtain indexes of top i contacts
        top_n_idxs = largest_indices(pred_contact_mtx, i)
        for contact in top_n_idxs:
            if contact in true_idxs:
                no_true_positives += 1
            else:
                no_false_positives += 1
        tpr = no_true_positives / no_positives
        tpr_per_rank.append(tpr)
        ppv = no_true_positives / (no_true_positives + no_false_positives)
        ppv_per_rank.append(ppv)

    return tpr_per_rank, ppv_per_rank


def largest_indices(array, n):
    """
    Returns the n largest indices from a numpy array.
    """
    if n > 0 and n <= array.size:
        flat = array.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        top_n_idxs = np.unravel_index(indices, array.shape)
        top_n_idxs = list(zip(*top_n_idxs))
    elif n <= 0:
        raise ValueError('n must be greater than 0')
    elif n > array.size:
        raise ValueError('n must be smaller or equal to the size of the array')
    return top_n_idxs


def evaluate_contact_predictions(true_contact_mtx, pred_contact_mtx):
    """
    Given the ground truth on the contact matrix and the predictions, calculate
    the Matthews Correlation Coefficient of the contact predictions.
    Note that this calculation is purely binary: it does not take into account
    the strength of the coupling, only whether the coupling is greater than 0,
    (presence/absence of the coupling in the contact matrix taken as ground
    truth).

    Arguments
    ---------
    true_contact_mtx: array-like, ground truth contact matrix
    pred_contact_mtx: array-like, predicted contact matrix

    Returns
    -------
    mcc:              float, value of the Matthews Correlation Coefficient
    """
    # Discretize predictions
    for idxs, val in np.ndenumerate(pred_contact_mtx):
        if val > 0:
            pred_contact_mtx[idxs] = 1
        if val <= 0:
            pred_contact_mtx[idxs] = 0
    mcc = matthews_corrcoef(true_contact_mtx.flatten(),
                            pred_contact_mtx.flatten())
    return mcc


def discretize_pred_contact_mtx(pred_contact_mtx, contact_threshold=0):
    """
    Given a predicted contact matrix, discretize the values in it according to
    a specified threshold: everything above it will be considered a contact.
    This is done with purposes of evaluating prediction performance.

    Arguments
    ---------
    pred_contact_mtx: array-like, predicted contact matrix
    contact_threshold: float, threshold for discretizing the matrix

    Returns
    -------
    pred_contact_mtx: array-like, discretized predicted contact matrix
    """
    for idx, item in np.ndenumerate(pred_contact_mtx):
        if pred_contact_mtx[idx] > contact_threshold:
            pred_contact_mtx[idx] = 1
        else:
            pred_contact_mtx[idx] = 0
    return pred_contact_mtx

