# -*- coding: utf-8 -*-

"""Utility functions for clustering methods."""

import warnings
from pathlib import Path
from typing import Callable

import numpy as np


def cluster_param_binary_search(
        results_folder: Path,
        fasta_file: Path,
        sequence_ids: list[str],
        init_args: tuple,
        min_args: tuple,
        max_args: tuple,
        user_args: str,
        threads: int,
        trial: Callable,
        args2str: Callable,
        gen_args: Callable,
        log_dir: Path,
) -> tuple[list[str], dict[str, str], np.ndarray]:
    """
    Perform binary search on the parameter space for clustering algorithms. So far, this is used to find optimal number
    of clusters for CD-HIT and MMseqs2.

    Args:
        results_folder: folder containing input and output files
        fasta_file: FASTA file of sequences to cluster.
        sequence_ids : IDs of sequences
        init_args: Initial arguments for optimization.
        min_args: The lower bound for the arguments.
        max_args: The upper bound for the arguments.
        user_args: Additional arguments that the user may have provided.
        threads: Number of threads to be used by the clustering algorithm.
        trial: Callable method running the actual clustering algorithm.
        args2str: Convert arguments to string to include them in filenames.
        gen_args: A callable function that generates a new argument configuration for the binary search. Has to be
            callable with the parameters of the last two configurations.
        log_dir: Directory to store the logs.

    Returns:
        Return the cluster names, the mapping from names to cluster names, and a similarity or distance matrix

    """
    def args2log(x: tuple):
        """
        Compute the name of the log file based on the provided arguments.

        Args:
            x: Arguments used in the run we want to store the results for

        Returns:
            Path to the file to write the execution log to
        """
        user_str = ""
        if len(user_args):
            user_str = f"_{user_args.replace('-', '').replace(' ', '_')}"
        return None if not log_dir else log_dir / f"{args2str(x).replace('-', '').replace(' ', '_')}{user_str}.log"

    # cluster with the initial arguments
    cluster_names, cluster_map, cluster_sim = trial(
        results_folder,
        fasta_file,
        args2str(init_args),
        user_args,
        threads,
        args2log(init_args)
    )
    num_clusters = len(cluster_names)

    # there are too few clusters, rerun with maximal arguments which has to result in every sample becomes a cluster
    if num_clusters <= 10:
        min_args = init_args
        min_clusters = num_clusters
        min_cluster_names, min_cluster_map, min_cluster_sim = cluster_names, cluster_map, cluster_sim
        max_cluster_names, max_cluster_map, max_cluster_sim = sequence_ids, dict(
            (n, n) for n in sequence_ids), np.zeros((len(sequence_ids), len(sequence_ids)))
        max_clusters = len(max_cluster_names)

    # if the number of clusters ranges in a good window, return the result
    elif 10 < num_clusters <= 100:
        return cluster_names, cluster_map, cluster_sim

    # too many clusters have been found, rerun the clustering with minimal arguments to find the lower bound of clusters
    else:
        max_args = init_args
        max_clusters = num_clusters
        max_cluster_names, max_cluster_map, max_cluster_sim = cluster_names, cluster_map, cluster_sim
        min_cluster_names, min_cluster_map, min_cluster_sim = \
            trial(results_folder, fasta_file, args2str(min_args), user_args, threads, args2log(min_args))
        min_clusters = len(min_cluster_names)

    # if the minimal number of clusters is in the target window, return them
    if 10 < min_clusters <= 100:
        return min_cluster_names, min_cluster_map, min_cluster_sim

    # if the maximal number of clusters is in the target window, return them
    if 10 < max_clusters <= 100:
        return max_cluster_names, max_cluster_map, max_cluster_sim

    # if the maximal number of clusters is still less than the lower bound of the window, report and warn
    if max_clusters < 10:
        warnings.warn(f"{trial.__name__[:-6]} cannot optimally cluster the data. The maximal number of clusters is "
                       f"{max_clusters}.")
        return max_cluster_names, max_cluster_map, max_cluster_sim

    # if the minimal number of clusters is still more than the upper bound of the window, report and warn
    if 100 < min_clusters:
        warnings.warn(f"{trial.__name__[:-6]} cannot optimally cluster the data. The minimal number of clusters is "
                       f"{min_clusters}.")
        return min_cluster_names, min_cluster_map, min_cluster_sim

    # for 8 rounds, apply binary search on the variable parameter space and try to hit the target window
    iteration_count = 0
    while True:
        iteration_count += 1
        args = gen_args(min_args, max_args)
        cluster_names, cluster_map, cluster_sim = trial(results_folder, fasta_file, args2str(args),
                                                        user_args, threads, args2log(args))
        num_clusters = len(cluster_names)
        if num_clusters <= 10 and iteration_count < 8:
            min_args = args
        elif 10 < num_clusters <= 100 or iteration_count >= 8:
            return cluster_names, cluster_map, cluster_sim
        else:
            max_args = args
