# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Callable
from typing import List
from enum import Enum, auto as enum_auto

import numpy as np
from pulp import *
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.SimDivFilters import rdSimDivPickers

from ...clustering.interface import MoleculeClusteringMethod



class RandomClustering(MoleculeClusteringMethod):

    """
    Randomly cluster a list of SMILES strings into n_clusters clusters.
    
    Attributes
    ----------
    n_clusters : int, optional
        Number of clusters.
    seed : int, optional
        Random seed.
    """

    def __init__(self, n_clusters : int = None, seed : int = 42) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.seed = seed

    def __call__(self, string_list : List[str]) -> dict:
        """
        Randomly cluster a list of SMILES strings into n_clusters clusters.
        
        Parameters
        ----------
        string_list : list[str]
            List of SMILES strings to cluster.
        
        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices and values are indices of SMILES strings.
        """

        self._set_n_clusters(len(string_list))

        # Initialize clusters
        clusters = { i: [] for i in range(self.n_clusters) }

        # Randomly assign each molecule to a cluster
        indices = np.random.RandomState(seed=self.seed).permutation(len(string_list))
        for i, index in enumerate(indices):
            clusters[i % self.n_clusters].append(index)

        return clusters

        
class MurckoScaffoldClustering(MoleculeClusteringMethod):

    """
    Cluster a list of SMILES strings based on Murcko scaffolds.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, string_list : List[str]) -> dict:

        """
        Cluster a list of SMILES strings based on Murcko scaffolds.

        Parameters
        ----------
        string_list : list[str]
            List of SMILES strings to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices and values are indices of SMILES strings.
        """
            
        # Generate scaffolds for each molecule
        mols = [Chem.MolFromSmiles(smiles) for smiles in string_list]
        scaffolds = [ MurckoScaffold.GetScaffoldForMol(mol) for mol in mols ]

        # Get unique scaffolds and initialize clusters
        unique_scaffolds = list(set(scaffolds))
        clusters = { i: [] for i in range(len(unique_scaffolds)) }

        # Cluster molecules based on scaffolds
        for i, scaffold in enumerate(scaffolds):
            clusters[unique_scaffolds.index(scaffold)].append(i)

        return clusters


class MoleculeSimilarityClustering(MoleculeClusteringMethod):
    
        """
        Abstract base class for clustering methods based on molecular dissimilarity.
        """
    
        def __init__(self, fp_calculator : Callable = GetMorganGenerator(radius=3, fpSize=2048) ) -> None:
            super().__init__()
            self.fp_calculator = fp_calculator

        def __call__(self, string_list : List[str]) -> dict:

            """
            Cluster a list of SMILES strings based on molecular dissimilarity.
            
            Parameters
            ----------
            string_list : list[str]
                List of SMILES strings to cluster.
            
            Returns
            -------
            clusters : dict
                Dictionary of clusters, where keys are cluster indices and values are indices of SMILES strings.
            """
            # TODO: support sparse fingerprints
            fps = [self.fp_calculator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in string_list]

            # Get cluster centroids and initialize clusters
            centroid_indices = self._get_centroids(fps)
            clusters = { i: [] for i in range(len(centroid_indices)) }

            # Cluster molecules based on centroids
            for i, fp in enumerate(fps):
                similarities = [DataStructs.FingerprintSimilarity(fp, fps[j]) for j in centroid_indices]
                clusters[np.argmax(similarities)].append(i)

            return clusters
    
        @abstractmethod
        def _get_centroids(self, fps : list) -> list:
            pass


class MaxMinClustering(MoleculeSimilarityClustering):

    """
    Cluster a list of SMILES strings based on molecular dissimilarity using the MaxMin algorithm.

    Attributes
    ----------
    fp_calculator : Callable, optional. 
        Function to compute molecular fingerprints.
    n_clusters : int, optional
        Number of clusters.
    seed : int, optional
        Random seed.
    """

    def __init__(
            self, 
            fp_calculator : Callable = GetMorganGenerator(radius=3, fpSize=2048),
            n_clusters : int = None,
            seed : int = 42,
        ) -> None:
        super().__init__(fp_calculator)
        self.n_clusters = n_clusters
        self.seed = seed

    def _get_centroids(self, fps : list) -> list:

        """
        Get cluster centroids using the MaxMin algorithm.
        
        Parameters
        ----------
        fps : list
            List of molecular fingerprints.
        
        Returns
        -------
        centroid_indices : list
            List of indices of cluster centroids.
        """

        self._set_n_clusters(len(fps))

        picker = rdSimDivPickers.MaxMinPicker()
        centroid_indices = picker.LazyBitVectorPick(fps, len(fps), self.n_clusters, seed=self.seed)

        return centroid_indices


class LeaderPickerClustering(MoleculeSimilarityClustering):

    """
    Cluster a list of SMILES strings based on molecular dissimilarity using LeadPicker to select centroids.

    Attributes
    ----------
    fp_calculator : Callable, optional.
        Function to compute molecular fingerprints.
    similarity_threshold : float, optional.
        Similarity threshold for clustering.
    """

    def __init__(
            self, 
            fp_calculator: Callable = GetMorganGenerator(radius=3, fpSize=2048),
            similarity_threshold : int = 0.7
     ) -> None:
        super().__init__(fp_calculator)
        self.similarity_threshold = similarity_threshold

    def _get_centroids(self, fps : list) -> list:

        """
        Get cluster centroids using LeadPicker.

        Parameters
        ----------
        fps : list
            List of molecular fingerprints.
        
        Returns
        -------
        centroid_indices : list
            List of indices of cluster centroids.
        """

        picker = rdSimDivPickers.LeaderPicker()
        centroid_indices = picker.LazyBitVectorPick(fps, len(fps), self.similarity_threshold)

        return centroid_indices
