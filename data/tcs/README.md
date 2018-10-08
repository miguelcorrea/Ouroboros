# README
This folder contains the TCS dataset.

* contact_mtx_8angstrom.csv is the used contact matrix
* aln_hk_100int.fasta and aln_rr_100int.fasta contain the interacting set; a line in one file corresponds to an interacting protein with the sequence at the same line in the other file

The mixed folder contains alignments ready for use by the algorithm, with different proportions of interacting sequences. As an indication of running time, running one of the examples with 75% interacting proteins on 20 Intel Xeon cores (CPU E5-2640 v3 @ 2.60GHz) took 226 CPU hours.