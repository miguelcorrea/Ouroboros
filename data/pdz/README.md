# README
This folder contains PDZ-peptide related data.

* specificity.csv contains information about the different specificity classes
* contact_mtx_8angstrom.csv is the used contact matrix
* matched_peptides_stringent.fasta and matched_domains_stringent.fasta contain the interacting set; a line in one file corresponds to an interacting protein with the sequence at the same line in the other file
* random_domains_stringent_new_X and random_peptides_stringent_new_X contain different non-interacting sets, followed by the random seed used

The mixed folder contains alignments ready for use by the algorithm, with different proportions of interacting sequences.
As an indication of running time, running one of the examples with 75% interacting proteins on 20 Intel Xeon cores (CPU E5-2640 v3 @ 2.60GHz) took 32 CPU hours.