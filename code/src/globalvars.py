#!/usr/bin/python
"""
Global variables
"""
from numpy import logspace, log10

global LOGO
LOGO = r"""
# Ouroboros | v 0.2 (October 2018)
# www.bif.wur.nl
# Distributed under the BSD-3 License
# - - - - - - - - - - - - - - - - - -
"""

global END
END = r"""
=================================================================
Analysis finished!
If you use this software in your research, please cite our paper:
https://doi.org/10.1101/254789
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
# list(np.logspace(-3,np.log10(1),15)) + [10,20,30,40]
global ALPHA_RANGE
# 15 12 9 6 3
ALPHA_RANGE = list(logspace(-3, log10(1), 3)) + [10, 20, 30, 40]

# ALPHA_RANGE = [0.001,
#                0.0016378937069540646,
#                0.0026826957952797246,
#                0.0043939705607607907,
#                0.0071968567300115215,
#                0.011787686347935873,
#                0.019306977288832496,
#                0.031622776601683791,
#                0.0517947467923121,
#                0.084834289824407175,
#                0.13894954943731375,
#                0.22758459260747887,
#                0.37275937203149379,
#                0.61054022965853261,
#                1.0, 10, 20, 30, 40]
