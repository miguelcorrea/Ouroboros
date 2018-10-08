Ouroboros
==============
[![DOI](https://zenodo.org/badge/119078766.svg)](https://zenodo.org/badge/latestdoi/119078766)

This is the repository for Ouroboros, an implementation of an algorithm for intermolecular coevolutionary analysis. Ouroboros is meant to be used when the multiple sequence alignments to be analyzed contain pairs of non-interacting sequences, which decreases intermolecular contact prediction performance. Ouroboros attempts to identify, without any prior information, which pairs are interacting and which are not. These protein-protein interaction predictions can in turn be used to improve contact prediction performance.

# FAQ

## How does it work?
Ouroboros combines coevolutionary analysis with expectation-maximization to simultaneously model protein-protein interaction and intermolecular contacts. It alternates between two steps: predicting protein-protein interaction based on the model describing intermolecular contacts, and estimating the model parameters based on protein-protein interaction predictions. By weighting proteins in the models according to the protein-protein interaction predictions, we boost the coevolutionary signal. For further details, please [read our paper](https://www.biorxiv.org/content/early/2018/01/28/254789).

## How do I run it?
The *run_analysis.py* script takes only one argument, the path to a JSON file that contains options for the arguments. Some examples of these can be found in data/params.
It is required that the file contains at least:
* io: path where results will be saved
* msa1: path to the first MSA
* msa2: path to the second MSA
* init: how to initialize the model: either "warm" or "random"
* mode: whether to perform "soft" or "hard" EM
* int_frac: prior fraction of interacting proteins (between 0 and 1)

If you know the ground truth about your input and want to test performance, you will also need to pass:
* test: set to True
* int_limit: index of the last interacting sequence pair in your dataset. Please prepare the MSAs so that the interacting sequences are at the top, and the non-interacting ones at the bottom. 
* contact_mtx: (optional) path to the contact matrix in CSV format. There should be as many rows as columns in msa1, and as many columns as in msa2.

As additional options, you can set:
* n_jobs: number of CPUs to use in model fitting (default: 2)
* n_starts: number of random starts to perform (if using a "random" start, ignored otherwise; default: 5)
* dfmax: maximum number of degrees of freedom allowed in model fitting (default:100)
* gap_threshold: gap frequency threshold; MSA columns with a gap frequency above this threshold are deleted (default: 0.5)
* tol: tolerance of the algorithm in soft EM, below which we consider the algorithm converges (default:0.005)
* max_iters: maximum number of EM iterations to perform (default:20)
* max_init_iters: maximum number of logistic regression iterations during model initialization (default: 100)
* max_reg_iters: maximum number of logistic regression iterations after initialization (default: 750)
* predict_contacts: boolean, whether to perform a final round of contact prediction or not. Although we have not focused in this area, and it is not extensively tested, it is possible obtain contact predictions.

## How do I interpret the output?
The most important files to look at in the results are:
* labels_per_iter.csv is a matrix (of dimensions sequence pairs x EM iterations) containing the interaction probability for each sequence pairs over iterations of EM
* output/z_over_iters.pdf shows the evolution of the protein-protein interaction probabilities over EM iterations
* output/convergence.png shows the evolution of the log-likelihoods

If you have passed ground truth about the interactions, you will obtain some additional files:
* true_alt_llhs.csv contains the log-likelihood of the ground truth according to the coevolutionary model over EM iterations
* true_null_llhs.csv contains the log-likelihood of the ground truth according to the independent model over EM iterations
* true_total_llhs.csv contains the total log-likelihood of the ground truth over EM iterations
* output/model_report.txt contains the model performance (as cross-entropy log-loss and Matthews Correlation Coefficient).
* output/conf_mtx.png and norm_conf_mtx.png contain confusion matrices; the second one is normalized to take into account class imbalance.
* output/perf_per_iter.png shows the evolution of the performance over EM iterations

Some additional files provide more detailed information:
* all_alt_llhs.csv contains the log-likelihood of the data according to the coevolutionary model over EM iterations
* all_null_llhs.csv contains the log-likelihood of the data according to the the independent evolution model over EM iterations
* all_total_llhs.csv contains the total log-likelihood of the data over EM iterations
* num_mtx_*.csv contains the alignments as numeric matrices  
* bin_mtx_*.csv contains the one-hot encoded alignments (binary matrices in the paper)
* processed_contact_matrix.csv (if you have passed a contact matrix) contains the ground truth contact matrix once it has been processed like the input alignments (i.e. removal of constant and gappy columns).  
* output/alt_llhs_mtx_*.csv contains the log probability of each individual residue at the concatenated alignments at a particular iteration according to the coevolutionary model
* output/null_llhs_mtx_*.csv contains the log probability of each individual residue at the concatenated alignments at a particular iteration according to the null model
* output/fixed_alphas_*_iter_init.csv contains the selected regularization strengths for each column for MSA A and B.
* output/dfs_*_iter_init.csv contains the degrees of freedom of the models selected during initialization

If you choose to perform contact prediction, you will obtain some additional output:
* output/final_contact_mtx.csv is a matrix (of dimensions msa1 x msa2, once they have been processed) containing coevolutionary strengths between all pairs of residues between your two MSAs
* output/norm_final_contact_mtx.csv is the same matrix, normalized using the Average Product Correction of Dunn *et al.* (2008)

Other undocumented output is present for development and testing reasons, and might be removed in the future.

## A warning tells me the log-likelihood is not increasing monotonically.
In hard EM, the log-likelihood is not mathematically guaranteed to increase monotonically, and so this does not necessarily indicate a problem. In soft EM, however, it is theoretically supposed to. We have observed that, in rare cases, and when the log-likelihood of the data has plateaued, one of the logistic regression models does not estimate the parameters well enough, leading to small decreases in log-likelihood. Although we do not expect such small decreases to negatively affect the results, if this happens, you should inspect your results, especially the evolution of the log-likelihood. This issue can be solved by simply increasing the maximum number of iterations of logistic regression (i.e. the max_reg_iters parameter) in order to improve estimation of model coefficients.

## How do I cite it?
If you use our software in your research, please cite our paper:
"Improving intermolecular contact prediction through protein-protein interaction prediction using coevolutionary analysis with expectation-maximization." bioRxiv (2018): 254789

# Contributing
We welcome contributions to improve our software, including feature additions, unit tests, bug fixes and improvements to documentation. 

# License
This is an open source tool available under the BSD-3 license. See the LICENSE file for details.

# Important links
* Source code repo: https://github.com/miguelcorrea/Ouroboros
* Issue tracker: https://github.com/miguelcorrea/Ouroboros/issues
* Preprint: https://www.biorxiv.org/content/early/2018/01/28/254789

# Requirements
* Python 3.6+
* NumPy
* scikit-learn
* scikit-bio
* matplotlib
* seaborn
* tqdm
* biopython
