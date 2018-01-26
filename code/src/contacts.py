"""
Module containing functions for intermolecular contact prediction.

@author: Miguel Correa
"""

import numpy as np
import subprocess

from collections import OrderedDict
from sklearn.metrics import matthews_corrcoef
from Bio import SeqIO
from skbio import TabularMSA, Protein

import warnings

global THREE_TO_ONE
THREE_TO_ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N',
                'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
                'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W',
                'TYR': 'Y', 'VAL': 'V'}

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


##########################################
# Distance / contact matrix calculations #
##########################################

def remove_water(chain):
    """
    Given a protein chain, remove waters forming part of the sequence.

    Arguments
    ---------
    chain:  Bio.PDB.Chain.Chain object

    Returns
    -------
    filtered_chain: list of Bio.PDB.Residue.Residue, with no "water residues"
    """
    filtered_chain = []
    for res in list(chain):
        if res.__dict__['resname'] == 'HOH':
            pass
        else:
            filtered_chain.append(res)
    return filtered_chain


def three_letter_to_one_letter(chain, code=THREE_TO_ONE):
    """
    'Translate' a chain of amino acids in three letter code to one letter code.

    Arguments
    ---------
    chain: Bio.PDB.Chain.Chain object or list of Bio.PDB.Residue.Residue
    code:  dict, conversion table

    Returns
    -------
    translated_chain: string
    """
    translated_chain = []
    for res in list(chain):
        try:
            translated_chain.append(THREE_TO_ONE[res.__dict__['resname']])
        except KeyError:
            warnings.warn('Unknown amino acid encountered: {}, skipping'.format(
                res), RuntimeWarning)
    return ''.join(translated_chain)


def calc_residue_distance(residue_one, residue_two):
    """
    Given two residues, calculate the Euclidean distance between them.
    The distance is measured between the beta carbons (or, in the case of
    glycine, with respect to the alpha carbon).

    Arguments
    ---------
    residue_one: Bio.PDB.Residue.Residue object
    residue_two: Bio.PDB.Residue.Residue object

    Returns
    -------
    dist:  float, Euclidean distance between the two residues
    """
    is_one_glycine = (residue_one.__dict__['resname'] == 'GLY')
    is_two_glycine = (residue_two.__dict__['resname'] == 'GLY')

    if is_one_glycine and is_two_glycine:
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    elif is_one_glycine and not is_two_glycine:
        diff_vector = residue_one["CA"].coord - residue_two["CB"].coord
    elif not is_one_glycine and is_two_glycine:
        diff_vector = residue_one["CB"].coord - residue_two["CA"].coord
    else:
        diff_vector = residue_one["CB"].coord - residue_two["CB"].coord

    dist = np.sqrt(np.sum(diff_vector * diff_vector))
    return dist


def calc_dist_matrix(chain_one, chain_two):
    """
    Given two proteins chains from a PDB file, calculate a matrix containing
    all pairwise Euclidean distances between residues from the two different
    chains.

    Arguments
    ---------
    chain_one, chain_two:  Bio.PDB.Chain.Chain object, or
                           list of Bio.PDB.Residue.Residue

    Returns
    -------
    dist_mtx:   array-like, matrix containing pairwise Euclidean distances
                between residues, of dimensions (chain_one, chain_two)
    """
    dist_mtx = np.zeros((len(chain_one), len(chain_two)))
    for i, residue_one in enumerate(chain_one):
        for j, residue_two in enumerate(chain_two):
            dist_mtx[i, j] = calc_residue_distance(residue_one, residue_two)
    return dist_mtx


def get_contact_matrix(dist_mtx, threshold):
    """
    Discretize distance matrix according to a given threshold.
    Values in the matrix that are equal or below the threshold will be True,
    otherwise they are False.
    This is done with the purpose of creating a contact map.

    Arguments
    ---------
    dist_mtx:   array-like, matrix containing pairwise Euclidean distances
                between residues, of dimensions (chain_one, chain_two)
    Returns
    -------
    contact_mtx: array-like, Boolean matrix; entries with a True value
                 represent a contact
    """
    return dist_mtx <= threshold

#########################################
# Mapping structure and contacts to MSA #
#########################################


def read_seqs_from_pdb(pdb_path):
    """
    Extract sequences from PDB file and remove duplicates.

    Arguments
    ---------
    pdb_path: string, path to PDB file

    Returns
    -------
    seqs:   list of strings, contains sequences present in the PDB file
    """
    seqs = []
    with open(pdb_path, "r") as source:
        for record in SeqIO.parse(source, 'pdb-seqres'):
            seqs.append(str(record.seq))
    seqs = list(set(seqs))
    return seqs


def align_new_seqs(existing_aln_path, new_seqs_path, out_path):
    """
    Add new sequences with an alignment using MAFFT.
    Returns None, writes a file to disk.

    Arguments
    ---------
    existing_aln_path: str, path to alignment file
    new_seqs_path:     str, path to file containing new sequences to add to the
                       alignment
    out_path:          str, path to output file

    """
    #
    args = ['mafft', '--add', new_seqs_path, '--keeplength', existing_aln_path]
    with open(out_path, 'w') as target:
        return_code = subprocess.call(args, stdout=target)
    if return_code != 0:
        raise Exception(
            'MAFFT finished with error code {}'.format(return_code))


##########################################################
# Mapping contact matrix to multiple sequence alignments #
##########################################################


def map_contact_mtx_to_alignment(contact_mtx, orig_seq_one, orig_seq_two,
                                 aln_seq_one, aln_seq_two):
    """
    Function to map a contact matrix derived from PDB files to aligned protein
    sequences. Put another way, given a contact matrix of dimensions
    corresponding to the lengths of the proteins in the PDB file, this
    function returns a contact matrix adjusted to the size of the same proteins
    once they have been aligned, with the contacts in the corresponding
    positions.

    Arguments
    ---------
    contact_mtx:    array-like, contact matrix derived from the PDB file, of
                    dimensions (orig_seq_one, orig_seq_two)
    orig_seq_one:   str, protein sequence extracted from the PDB file
    orig_seq_two:   str, protein sequence extracted from the PDB file
    aln_seq_one:    str, protein sequence orig_seq_one once it's been aligned
    aln_seq_two:    str, protein sequence orig_seq_two once it's been aligned

    Return
    ------
    mapped_contact_mtx: array-like, contact matrix of dimensions
                        (aln_seq_one, aln_seq_two)
    """
    # Map sequences extracted from PDB file to aligned sequences
    seq_map_one = map_seq_to_alignment(orig_seq_one, aln_seq_one)
    seq_map_two = map_seq_to_alignment(orig_seq_two, aln_seq_two)
    # Get from the contact matrix which residues are in contact
    # for both sequences
    mapped_coords = map_contacts_to_alignment(
        contact_mtx, seq_map_one, seq_map_two)
    # Build a contact matrix adjusted to the dimensions of the aligned proteins
    mapped_contact_mtx = reconstruct_contact_mtx(
        mapped_coords, aln_seq_one, aln_seq_two)

    return mapped_contact_mtx


def map_seq_to_alignment(orig_seq, aln_seq):
    """
    Function to identify which positions in a sequence correspond to positions
    in the same sequence once it is aligned.

    Arguments
    ---------
    orig_seq: str, unaligned sequence
    aln_seq:  str, aligned sequence

    Returns
    -------
    seq_map: OrderedDict, in the format
             {(index in original sequence, index in aligned sequence)}

    """
    seq_map = OrderedDict()
    # Search each residue in the original sequence in the aligned sequence
    for i, res_one in enumerate(orig_seq):
        for j, res_two in enumerate(aln_seq):
            # They have to be the same residue and not have been mapped before
            if res_one == res_two and j not in seq_map.values():
                seq_map[i] = j
                break
    keys = seq_map.keys()
    vals = seq_map.values()
    # Check that the list of keys is sequential
    assert list(keys) == list(range(min(keys), max(keys) + 1))
    # Check that there are no duplicates in the values and that the ordering
    # is sequential
    assert list(set(vals)) == list(vals)
    return seq_map


def get_contacts_from_matrix(contact_mtx):
    """
    Retrieve from the contact matrix the indexes at which there are contacts
    for the two sequences.

    Arguments
    ---------
    contact_mtx: array-like, contact matrix

    Returns
    -------
    contact_coords: list, contains tuples of indexes (i,j) indicating the
                    positions in the contact matrix where there are contacts
    """
    contact_coords = []
    for idx, val in np.ndenumerate(contact_mtx):
        if val == 1:
            contact_coords.append(idx)

    return contact_coords


def map_contacts_to_alignment(contact_mtx, seq_map_one, seq_map_two):
    """
    Given a contact matrix, find what positions in the aligned sequences
    correspond to the contacts.

    Arguments
    ---------
    contact_mtx: array-like, contact matrix derived from the PDB file, of
                 dimensions (orig_seq_one, orig_seq_two)
    seq_map:     OrderedDict, in the format
                 {(index in original sequence, index in aligned sequence)}

    Returns
    -------
    mapped_coords: list of tuples, where each tuple (i,j) indicates a pair
                   of positions that make contact
    """
    contact_coords = get_contacts_from_matrix(contact_mtx)
    mapped_coords = []
    for contact in contact_coords:
        pos_one = seq_map_one[contact[0]]
        pos_two = seq_map_two[contact[1]]
        mapped_coords.append((pos_one, pos_two))
    return mapped_coords


def reconstruct_contact_mtx(mapped_coords, aln_seq_one, aln_seq_two):
    """
    Given indexes for the contacts corresponding to positions in the multiple
    sequence alignments, build a contact matrix of appropiate dimensions.

    Arguments
    ---------
    mapped_coords: list of tuples, where each tuple (i,j) indicates a pair
                   of positions that make contact
    aln_seq_one:    str, protein sequence orig_seq_one once it's been aligned
    aln_seq_two:    str, protein sequence orig_seq_two once it's been aligned

    Returns
    -------
    mapped_contact_mtx: array-like, contact matrix of dimensions
                        (aln_seq_one, aln_seq_two)
    """
    mapped_contact_mtx = np.zeros((len(aln_seq_one), len(aln_seq_two)))
    for contact in mapped_coords:
        mapped_contact_mtx[contact] = 1
    return mapped_contact_mtx


def postprocess_contact_mtx(contact_mtx, aln_one_path, aln_two_path):
    """
    Remove from the contact matrix positions that would also be removed from
    the alignments because they contain too many gaps or are constant.

    Arguments
    ---------
    contact_mtx: array-like, contact matrix derived from the PDB file, of
                 dimensions (orig_seq_one, orig_seq_two)
    aln_one_path: str, path to multiple sequence alignment
    aln_two_path: str, path to multiple sequence alignment

    Returns
    -------
    processed_contact_mtx: array-like
    """
    # Find gappy columns
    msa_a = TabularMSA.read(aln_one_path, constructor=Protein)
    msa_b = TabularMSA.read(aln_two_path, constructor=Protein)

    # Find constant columns
    constant_idxs_a = find_constant_columns(msa_a)
    constant_idxs_b = find_constant_columns(msa_b)
    # Find gappy columns
    gappy_idxs_a = find_gappy_columns(msa_a)
    gappy_idxs_b = find_gappy_columns(msa_b)

    del_idxs_a = list(set(constant_idxs_a + gappy_idxs_a))
    del_idxs_b = list(set(constant_idxs_b + gappy_idxs_b))

    # Remove those elements
    processed_contact_mtx = np.delete(contact_mtx, del_idxs_a, axis=0)
    processed_contact_mtx = np.delete(
        processed_contact_mtx, del_idxs_b, axis=1)
    return processed_contact_mtx


def find_constant_columns(msa):
    """
    Given a multiple sequence alignment, return indexes of columns that stay
    constant.
    """
    msa = [str(seq) for seq in iter(msa)]
    msa = list(zip(*msa))

    # Collect indices of columns that stay constant
    constant_idxs = []
    for idx, row in enumerate(msa):
        if len(set(row)) == 1:
            constant_idxs.append(idx)

    return constant_idxs


def find_gappy_columns(msa, gap_threshold=0.5):
    """
    Given a multiple sequence alignment, return indexes of columns that have
    a frequency of gaps equal to or above a certain threshold.
    """
    msa = [str(seq) for seq in iter(msa)]
    msa = list(zip(*msa))  # Transpose MSA for easier iteration

    # Collect indices of rows (columns in the original MSA) that have a gap
    # frequency equal or above to the treshold
    gappy_idxs = []
    for idx, row in enumerate(msa):
        if row.count('-') / len(row) >= gap_threshold:
            gappy_idxs.append(idx)

    return gappy_idxs
