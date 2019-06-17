"""
Module containing functions for visualization

@author: Miguel Correa Marrero
"""

import matplotlib as mpl
mpl.use("Agg")  # Allows plotting without a running X server
import matplotlib.pyplot as plt
plt.interactive(False)

import numpy as np
import seaborn as sns

import os
import math
import warnings

from sklearn.metrics import confusion_matrix, matthews_corrcoef, log_loss

from helpers import round_labels
from contacts import evaluate_contact_predictions, discretize_pred_contact_mtx


def draw_contact_mtx(data, file_path, cmap='magma'):
    """
    """
    sns.heatmap(data, cmap=cmap, vmin=0, vmax=np.max(data), annot=True,
                linewidths=.5, cbar_kws={'label': 'Coupling strength'})
    plt.xlabel('Positions in MSA B')
    plt.ylabel('Positions in MSA A')
    plt.title('Contact map')
    plt.savefig(file_path, bbox_inches='tight', format='png', dpi=1200)
    plt.clf()


def draw_label_heatmap(data, file_path, cmap="Greys_r"):
    """
    Draw a heatmap showing the evolution of the hidden variables accros
    iterations and save it to disk.

    Arguments
    ---------
    data:       array-like, 2D
    cmap:       string, matplotlib colormap
    file_path:  string, path to which the plot will be saved

    """
    # plt.figure(figsize=(40,40))
    sns.heatmap(data, cmap=cmap, vmin=0, vmax=1)
    plt.xlabel("Iterations")
    plt.ylabel("Sequence")
    plt.savefig(file_path, bbox_inches="tight", format='pdf', dpi=1200)
    plt.clf()


def draw_llh_heatmap(data, file_path):
    """
    Draw a heatmap showing the evolution of the alternative log-likelihoods
    accros iterations, plus the null
    log-likelihoods, and write it to disk.
    """

    cmap = plt.get_cmap('magma')
    xticks = [str(x) for x in range(data.shape[1])][:-1] + ['Null']
    sns.heatmap(data, cmap=cmap, xticklabels=xticks)

    plt.ylabel("Sequence")
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()


def draw_alt_vs_null(alt_llhs, null_llhs, file_path):
    """
    Draw a scatterplot showing the alternative and null log-likelihood per
    sequence.
    """
    plt.scatter(range(len(alt_llhs)), alt_llhs, label=(
        'Sequence pair alternative log-likelihood'), alpha=0.7, color='orange')
    plt.scatter(range(len(null_llhs)), null_llhs, label=(
        'Sequence pair null log-likelihood'), alpha=0.7, color='blue')

    plt.xlabel('Sequence pair index')
    plt.ylabel('Log-likelihood')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()


def draw_llh_mtx(llh_mtx, file_path):
    """
    Draw a heatmap showing the log-likelihoods for both alignments.
    """
    cmap = plt.get_cmap('magma')
    sns.heatmap(llh_mtx, cmap=cmap, vmin=llh_mtx.min(), vmax=llh_mtx.max())

    plt.xlabel('MSA positions')
    plt.ylabel('Sequence pair index')
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()


def draw_llh_plot(alt_llh, null_llh, int_llh, nonint_llh, total_llh, file_path,
                  perfect_int_llh=None, perfect_nonint_llh=None,
                  perfect_total_llh=None):
    """
    Draw a plot displaying the evolution of the likelihoods over iterations.

    Arguments
    ---------
    alt_llh:    array-like, contains log-likelihoods of the alternative model
    null_llh:   array-like, contains log-likelihoods of the null model
    int_llh:    array-like, contains log-likelihoods of alternative model, but
                only for sequences that are putatively interacting
    nonint_llh: array_like, contains log-likelihoods of null model, but only
                for sequences that are putatively non-interacting
    total_llh:  array_like, contains the sum of the log-likelihood of int_llh
                and nonint_llh

    file_path:  string, path to which the plot will be saved
    """
    palette = sns.color_palette("husl", 5)
    plt.plot(alt_llh, label=("""Sum of sequence log-likelihoods
                             (alternative model)"""), color=palette[0],
             alpha=0.7)
    plt.plot(null_llh, label=("""Sum of sequence log-likelihood
                              (null model)"""), color=palette[1],
             alpha=0.7)
    plt.plot(int_llh, label=("""Sum of interacting sequence log-likelihood
                             (alternative model)"""), color=palette[2],
             alpha=0.7)
    plt.plot(nonint_llh, label=("""Sum of non-interacting sequence
        log-likelihood (null model)"""), color=palette[3],
             alpha=0.7)
    plt.plot(total_llh, label=("""Sum of interacting sequence log-likelihood
        (alternative model) \n and non-interacting sequence log-likelihood
         (null model)"""),
             color=palette[4], alpha=0.7)

    if perfect_int_llh is not None:
        perfect_palette = sns.hls_palette(5, l=.2, s=.9)

        plt.plot(perfect_int_llh, label=("""Sum of interacting sequence
             log-likelihood (perfect solution, alternative model)"""),
                 color=perfect_palette[2], alpha=0.7, ls='dashed')

        plt.plot(perfect_nonint_llh, label=("""Sum of non-interacting sequence
         log-likelihood (perfect solution, null model)"""),
                 color=perfect_palette[3], alpha=0.7, ls='dashed')

        plt.plot(perfect_total_llh, label=("""Sum of interacting sequence
         log-likelihood (perfect solution, alternative model)
         and non-interacting sequence log-likelihood
         (perfect solution, null, model)"""),
                 color=perfect_palette[4], alpha=0.7, ls='dashed')

    plt.plot()

    plt.xlabel('Iteration')
    plt.ylabel('Total log-likelihood')

    iter_range = list(range(len(alt_llh)))  # Avoid having non-integer x ticks
    plt.xticks(iter_range)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.title('Evolution of likelihoods over iterations')
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()


def make_confusion_matrices(true_labels, pred_labels, plots_dir):
    """
    """
    conf_mtx = confusion_matrix(true_labels, pred_labels)
    draw_confusion_matrix(conf_mtx, os.path.join(plots_dir, "conf_matrix.png"))
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = conf_mtx.astype(
        'float') / conf_mtx.sum(axis=1)[:, np.newaxis]
    draw_confusion_matrix(cm_normalized, os.path.join(
        plots_dir, "norm_conf_matrix.png"),
        title='Normalized confusion matrix')


def draw_confusion_matrix(conf_mtx, file_path, cmap='Blues',
                          title='Confusion matrix'):
    """
    Draw a confusion matrix.

    Arguments
    ---------
    conf_mtx:    numpy array
    file_path:   string, path to which the plot will be saved
    cmap:        string, matplotlib colormap
    title:       string, plot title
    """
    labels = ['Non-interacting', 'Interacting']
    sns.heatmap(conf_mtx, xticklabels=labels,
                yticklabels=labels, cmap=cmap, annot=True)
    plt.title(title)
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()


def draw_performance_per_iter(labels_per_iter, true_labels, mode, out_path):
    """
    Draw a plot showing the predictive performance per iteration.
    Writes the plot to disk.

    Arguments
    ---------
    labels_per_iter: array-like, contains the values of the hidden variables
                     for each iteration
    true_labels:     array-like, contains the ground truth
    mode:            str, "hard" of "soft" EM
    out_path:        str, path to save the plot to

    Returns
    -------
    None
    """
    if len(set(true_labels)) > 1:
        if mode == 'hard':
            mccs = []
            for labels in labels_per_iter:
                mccs.append(matthews_corrcoef(true_labels, labels))
            plt.plot(mccs, label=('Matthews Correlation Coefficient'))
        elif mode == 'soft':
            mccs = []
            log_losses = []
            for labels in labels_per_iter:
                mccs.append(matthews_corrcoef(
                    true_labels, round_labels(labels)))
                log_losses.append(log_loss(true_labels, labels))
            plt.plot(mccs, label=(
                """Matthews Correlation Coefficient (rounded labels"""))
            plt.plot(log_losses, label=('Cross-entropy log-loss'))

        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        iter_range = list(range(len(labels_per_iter)))
        plt.xticks(iter_range)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Evolution of predictive performance over iterations')
        plt.savefig(os.path.join(out_path, 'perf_per_iter.png'),
                    bbox_inches='tight')
        plt.clf()

    else:
        warnings.warn("""Ground truth contains only one class;
            not plotting performance per iteration""", RuntimeWarning)


