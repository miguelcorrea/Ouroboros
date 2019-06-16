#!/usr/bin/python
"""
Global variables
"""
from numpy import logspace, log10

global LOGO
LOGO = r"""
# Ouroboros | v 0.3 (June 2019)
# www.bif.wur.nl
# Distributed under the BSD-3 License
# - - - - - - - - - - - - - - - - - -
"""

global END
END = r"""
=================================================================
Analysis finished!
If you use this software in your research, please cite our paper:
https://doi.org/10.1093/bioinformatics/bty924
=================================================================
"""

##############################
#Amino acid conversion tables#
##############################

global AA_TABLE
AA_TABLE = {'A': 0, 'R': 1, 'N': 2,
            'D': 3, 'C': 4,
            'E': 5, 'Q': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10,
            'K': 11, 'M': 12, 'F': 13,
            'P': 14, 'S': 15, 'T': 16,
            'W': 17, 'Y': 18, 'V': 19,
            '-': 20}


#####################################
# Range of regularization strengths #
#####################################

global ALPHA_RANGE
ALPHA_RANGE = list(logspace(-3, log10(1), 15)) + [10, 20, 30, 40]
ALPHA_RANGE = list(reversed(sorted(ALPHA_RANGE)))
