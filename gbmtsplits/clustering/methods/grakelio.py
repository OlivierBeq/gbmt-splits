# -*- coding: utf-8 -*-

"""Converters from and to grakel graph objects."""


from __future__ import annotations

import math
from pathlib import Path

from Bio.PDB.Structure import Structure
from grakel import Graph
from Bio.PDB import PDBParser


from .. import Protein


def pdb_to_grakel(pdb: Path | GrakelPDBStructure | Protein, threshold: float = 7) -> Graph:
    """
    Convert a PDB file into a grakel graph to compute WLKs over them.

    Args:
        pdb: Either PDB structure or filepath to PDB file
        threshold: Distance threshold to apply when computing the graphs

    Returns:
        A grakel graph based on the PDB structure
    """
    if isinstance(pdb, Path):
        parser = PDBParser()
        structure = parser.get_structure(None, pdb)
        pdb = PDBStructure(pdb)

    tmp_edges = pdb.get_edges(threshold)
    edges = {}
    for start, end in tmp_edges:
        if start not in edges:
            edges[start] = []
        edges[start].append(end)

    return Graph(edges, node_labels=pdb.get_nodes())


class GrakelPDBStructure(Structure):
    """Extension of BioPython's PDB structure with edges and nodes for GraKel."""

    def __init__(self, structure: Structure):
        """Create a GrakelPDBStructure from a BioPython PDBStructure

        Parameters
        ----------
        structure : BioPython PDBStructure
        """
        super().__init__(structure.id)
        self.__dict__ = structure.__dict__

    def get_edges(self, thresh_dist: float = 7, only_calpha: bool = True) -> list[tuple[int, int]]:
        """Obtain edges of a PDB structure based on a distance threshold.

        Parameters
        ----------
        thresh_dist : float
            Distance threshold (in Angstroms) between two atoms to be considered an edge.
        only_calpha: bool
            Should only alpha carbons considered as nodes.

        Returns
        -------
        A list of edges given by their residue number
        """
        coords = [{'num': res.get_id()[1], 'coords': ([atom.get_coord().tolist()
                                               for atom in res.get_atoms()
                                               if atom.get_name() == 'CA'][0]
                                              if only_calpha
                                              else res.center_of_mass().tolist())}
                  for res in self.get_residues()
                  ]
        return [(coords[i]['num'], coords[j]['num'])
                for i in range(len(coords))
                for j in range(len(coords))
                if math.dist(coords[i]['coords'], coords[j]['coords']) < thresh_dist]

    def get_nodes(self) -> dict[int, int]:
        """
        Get the nodes as a map from their residue id to a numerical encoding of the represented amino acid.

        Returns:
            Dict mapping residue ids to a numerical encodings of the represented amino acids
        """
        aa_encoding = {
            "ala": 0, "arg": 1, "asn": 2, "asp": 3, "cys": 4, "gln": 5, "glu": 6, "gly": 7, "his": 8, "ile": 9,
            "leu": 10, "lys": 11, "met": 12, "phe": 13, "pro": 14, "ser": 15, "thr": 16, "trp": 17, "tyr": 18,
            "val": 19,
        }
        return dict(
            [(res.get_id()[1], (aa_encoding.get(res.get_resname().lower(),
                                               20)))
             for res in self.get_residues()])
