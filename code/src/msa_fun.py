#!/usr/bin/python
"""
Set of functions to manipulate multiple sequence alignments (MSAs).
@author: Miguel Correa
"""

import numpy as np
from skbio import TabularMSA, Protein


def del_gappy_cols(msa, gap_threshold):
    """
    Remove columns from a MSA where the frequency of gap occurrence is above a
    certain threshold

.   Arguments
    ----------
    msa: TabularMSA
        The MSA to be filtered
    gap_threshold: float
        Gap frequency threshold: columns that have a gap frequency equal or
        greater than this value will be removed.

    Returns
    -------
    msa: TabularMSA object
        Filtered MSA
    idxs: list of indexes of gappy columns
    """
    msa = [str(seq) for seq in iter(msa)
           ]  # TabularMSA format is not flexible enough yet
    # Transpose MSA for easier iteration
    msa = list(zip(*msa))

    # Collect indices of rows (columns in the original MSA) that have a gap
    # frequency equal or above to the treshold
    idxs = []
    for idx, row in enumerate(msa):
        if row.count('-') / len(row) >= gap_threshold:
            idxs.append(idx)
    # Remove gappy columns; always remove the one with the highest index!
    for idx in sorted(idxs, reverse=True):
        del msa[idx]
    # Return MSA to original orientation and in TabularMSA format for further
    # processing
    msa = list(zip(*msa))

    msa = TabularMSA([Protein(''.join(seq)) for seq in msa])
    return msa, idxs


def del_constant_cols(msa):
    """
    Remove columns where there is only one amino acid.

    Arguments
    ---------
    msa: array-like

    Returns
    -------
    msa: array-like
         Filtered MSA
    idxs: list of indexes of constant columns
    """
    msa = list(zip(*msa))

    # Collect indices of columns that stay constant
    idxs = []
    for idx, row in enumerate(msa):
        if len(set(row)) == 1:
            idxs.append(idx)
    # Delete constant columns
    for idx in sorted(idxs, reverse=True):
        del msa[idx]
    msa = list(zip(*msa))
    msa = np.array([list(item) for item in msa])

    return msa,idxs


def make_num_mtx(msa, aa_table):
    """
    Convert a multiple sequence alignment to a numeric matrix.
    The amino acids at each position of the alignment are converted into
    numeric factors according to a certain mapping.

    Arguments
    ----------
    msa: scikit-bio TabularMSA object
        The multiple sequence alignment to be transformed
    aa_table: dictionary
        Contains the mapping to convert amino acids into numeric factors

    Returns
    -------
    num_mtx: array-like, the numeric matrix

    """
    # Create an empty matrix
    seq_len = msa.shape[0]
    seq_cols = msa.shape[1]
    num_mtx = np.zeros((seq_len, seq_cols))

    i = 0

    # Fill the matrix by iterating over the alignment and mapping the values
    for seq in iter(msa):
        for pos in range(seq_cols):
            aa = str(seq)[pos]
            num_mtx[i, pos] = aa_table[aa]
        i += 1

    return num_mtx


def make_bin_mtx(num_mtx, aa_table):
    """
    Convert a numeric matrix into a binary matrix.
    Each column of the alignment occupies a submatrix that has as many columns
    as there are letters in the amino acid alphabet.

    If we consider an amino acid alphabet of 21 letters, take, for example the
    first 20 columns of the binary matrix. Row i corresponds to the ith
    sequence in the alignment, and column j to the jth amino acid in the
    alphabet. In this case, position [i,j] indicates if amino acid j is
    present in position 1 of the ith sequence. Gaps, the 21st symbol, are not
    an extra column in the binary matrix, as they can be predicted linearly
    from the other columns, and thus introduce collinearity.

    Arguments
    ----------
    num_mtx: array-like, the numeric matrix

    Returns
    -------
    bin_mtx: array-like, the binary matrix

    """
    # Initialize the binary matrix
    # Gaps will be represented by a vector of zeros; doing otherwise introduces
    # collinearity
    no_aas = len(aa_table.keys()) - 1
    mtx_rows = num_mtx.shape[0]
    mtx_cols = num_mtx.shape[1] * no_aas
    bin_mtx = np.zeros((mtx_rows, mtx_cols), dtype=np.int)

    # Fill the binary matrix
    offset = 0  # To keep track of which submatrix to fill
    for pos in range(num_mtx.shape[1]):
        num_col = num_mtx[:, pos]
        for row in range(len(num_col)):
            # Gaps are ignored, revise this if we use reduced alphabets
            if num_col[row] != 20:
                col = num_col[row] + offset
                bin_mtx[row, int(col)] = 1
        offset += no_aas

    return bin_mtx
