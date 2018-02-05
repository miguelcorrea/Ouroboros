Ouroboros
==============

This is the repository for Ouroboros, an implementation of an algorithm for intermolecular coevolutionary analysis. Ouroboros is meant to be used when the multiple sequence alignments to be analyzed contain pairs of non-interacting sequences, which decreases intermolecular contact prediction performance. Ouroboros attempts to identify, without any prior information, which pairs are interacting and which are not. These protein-protein interaction predictions can in turn be used to improve contact prediction performance.

# FAQ

## How does it work?
Ouroboros combines coevolutionary analysis with expectation-maximization to simultaneously model protein-protein interaction and intermolecular contacts. It alternates between two steps: predicting protein-protein interaction based on the model describing intermolecular contacts, and estimating the model parameters based on protein-protein interaction predictions. By weighting proteins in the models according to the protein-protein interaction predictions, we boost the coevolutionary signal. For further details, please read our paper.

## How do I run it?
The script takes only one argument, the path to a JSON file that contains options for the arguments.
It is required that the file contains at least:
* io: path where results will be saved
* msa1: path to one of the MSAs
* msa2: path to the other MSA
* init: how to initialize the model: either "warm" or "random"
* mode: whether to perform "soft" or "hard" EM
* int_frac: prior fraction of interacting proteins (between 0 and 1)

If you know the ground truth about your input, you will also need to pass:
* test: set to True
* int_limit: index of the last interacting sequence pair in your dataset. Please prepare the MSAs so that the interacting sequences are at the top, and the non-interacting ones at the bottom. 
* contact_mtx: (optional) path to the contact matrix in CSV format. There should be as many rows as columns in msa1, and as many columns as in msa2.
As additional options, you can set:
* n_jobs: number of CPUs to use in model fitting (default:2)
* n_starts: number of random starts to perform (if using a "random" start, ignored otherwise; default:5)
* dfmax: maximum number of degrees of freedom allowed in model fitting (default:100)
* tol: tolerance of the algorithm, below which we consider the algorithm converges (default:0.005)
* max_iters: maximum number of EM iterations to perform (default:20)

## How do I interpret the output?
* labels_per_iter.csv contains the interaction probability for each sequence pair (rows) over EM iterations (columns)
* all_int_llhs.csv contains the log-likelihood of the coevolutionary model over EM iterations
* all_null_llhs.csv contains the log-likelihood of the independent evolution model over EM iterations
* all_total_llhs.csv contains the total log-likelihood over EM iterations
If you have passed ground truth about the interactions, you will also get files with the prefix true- . These contain the log-likelihood of the true solution at each step.
The output folder contains some more detailed information:
* alt_llhs_mtx_*.csv files contain the matrix of probabilities according to the coevolutionary model, with the suffix indicating the EM iteration
* null_llhs_mtx_*.csv files contain the matrix of probabilities according to the independent evolution model, with the suffix indicating the EM iteration
* fixed_alphas_*_iter_init.csv contains the regularization strengths for each column for MSA A and B.
* z_over_iters.png shows the evolution of the protein-protein interaction probabilities
* convergence.png shows the evolution of the log-likelihoods
If you have passed the ground truth, a series of files detail the performance:
* model_report.txt contains the model performance (as cross-entropy log-loss and Matthews Correlation Coefficient).
* conf_mtx.png and norm_conf_mtx.png contain confusion matrices; the second one is normalized to take into account class imbalance.
* perf_per_iter.png shows the evolution of the performance

## I get a warning telling me the log-likelihood is not increasing monotonically.
In hard EM, the log-likelihood is not guaranteed to increase monotonically, and so this does not necessarily indicate a problem. In soft EM, however, it is theoretically supposed to. In rare occasions, however, the logistic regression might not estimate the parameters well enough. This leads to decreases in log-likelihood, although we do not expect it to happen with the current settings. This issue can be solved by increasing the maximum number of iterations of logistic regression.

## How do I cite it?
If you use our software in your research, please cite our paper:
Improving intermolecular contact prediction through protein-protein interaction prediction using coevolutionary analysis with expectation-maximization." bioRxiv (2018): 254789

# Contributing
We welcome contributions to improve our software, including feature additions, bug fixes and improvements to documentation. 

# License
This is an open source tool available under the BSD-3 license. See the LICENSE file for details.

# Important links
* Source code repo: https://github.com/miguelcorrea/Ouroboros
* Issue tracker: https://github.com/miguelcorrea/Ouroboros/issues

# Requirements
* Python 3
* NumPy
* scikit-learn
* scikit-bio
* matplotlib
* seaborn
* tqdm
