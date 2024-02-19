# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List


class ClusteringMethod(ABC):

    """
    Abstract base class for clustering methods.
    """
    def __init__(self):
        self.n_clusters = None

    @abstractmethod
    def __call__(self, string_list : List[str]) -> dict:
        """Cluster a list of protein sequences based on the initialized algorithm.

        Parameters
        ----------
        string_list : list[str]
            List of string representations of objects to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of objects' string representations.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _set_n_clusters(self, N : int) -> None:
        self.n_clusters = self.n_clusters if self.n_clusters is not None else N // 10


class MoleculeClusteringMethod(ClusteringMethod, ABC):
    pass


class ProteinClusteringMethod(ClusteringMethod, ABC):

    @abstractmethod
    def is_installed(self):
        pass


class ProteinSequenceClusteringMethod(ProteinClusteringMethod, ABC):
    pass


class ProteinCrystalStructureClusteringMethod(ProteinClusteringMethod, ABC):
    pass


class AlgorithmProteinClustering(Enum):
    CDHIT = enum_auto()
    FOLDSEEK = enum_auto()
    MASH = enum_auto()
    MMSEQS2 = enum_auto()
    MMSEQS2plusplus = enum_auto()
    TMALIGN = enum_auto()
    WLK = enum_auto()


class AlgorithmMoleculeClustering(Enum):
    RANDOM = enum_auto()
    MURCKO_SCAFFOLD = enum_auto()
    MAX_MIN = enum_auto()
    SPHERE_EXCLUSION = enum_auto()
    WLK = enum_auto()
