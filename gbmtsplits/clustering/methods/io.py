# -*- coding: utf-8 -*-

"""IO functions for clustering methods."""

import os
import tempfile
from io import StringIO
from pathlib import Path

from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.SeqRecord import Seq, SeqRecord

from .. import Protein


def sequences_to_fasta(sequences: list[Protein], dir: Path, name: str = None) -> Path:
    """
    Write protein sequences to a FASTA file.

    Parameters
    ----------
    sequences : list[Protein]
        Protein objects with sequences.
    dir : Path
        path of the folder containing the output FASTA files.
    name : str
        name of the file (without extension) if not temporary
    Returns
    -------
        path to the temporary FASTA file.
    """
    if not all(protein.sequence is not None for protein in sequences):
        raise ValueError('FASTA IO requires all Protein objects to have a sequence')
    seqs = [SeqRecord(seq=Seq(protein.sequence), id=protein.id, name='', description='')
            for protein in sequences]
    tmp_path = mktempfile(dir, prefix=name, suffix='.fasta')
    SeqIO.write(seqs, tmp_path, 'fasta')
    return Path(tmp_path)


def structure_to_pdb(structure: Protein, dir: Path, name: str = None) -> Path:
    """
    Write protein structures to a temporary PDB file.

    Parameters
    ----------
    structure : Protein
        Protein crystal structure as either 'pdb' or 'mmcif'.
    dir : Path
        path of the folder containing the output PDB file.
    name : str
        name of the file (without extension) if not temporary
    Returns
    -------
        path to the temporary PDB file.
    """
    if not (structure.pdb is not None or structure.mmcif is not None):
        raise ValueError('PDB/mmCIF IO requires all Protein objects to have a crystal structure')
    parser = PDBParser()
    pdbio = PDBIO()
    if structure.pdb is not None:
        structure = parser.get_structure(structure.id, StringIO(structure.pdb))
        tmp_path = mktempfile(dir, prefix=(name if name else '{structure.id}'), suffix='.pdb')
    else:
        structure = parser.get_structure(structure.id, StringIO(structure.mmcif))
        tmp_path = mktempfile(dir, prefix=(name if name else '{structure.id}'), suffix='.cif')
    pdbio.set_structure(structure)
    pdbio.save(tmp_path)
    return Path(tmp_path)


def mktempfile(dir: Path, prefix: str = None, suffix: str = None) -> Path:
    """Return the path to a writeable file."""
    file = tempfile.mkstemp(dir=dir.as_posix(), prefix=prefix, suffix=suffix)
    os.close(file[0])
    return Path(file[1])


def mktempdir(dir: Path, prefix: str = None, suffix: str = None, ensure_new: bool = True) -> Path:
    """Create a directory or ensure it exists.

    Parameters
    ----------
    dir : Folder containing the new directory to create.
    prefix : Prefix of the folder name.
    suffix : Suffix of the folder name.
    ensure_new : Ensure the folder does not already exist.

    Returns
    -------
        Path to the directory.
    """
    dir = tempfile.mkdtemp(dir=dir.as_posix(), prefix=prefix, suffix=suffix)
    return Path(dir)
