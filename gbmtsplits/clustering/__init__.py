# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class Protein:
    id: str
    sequence: str = None
    pdb: str = None
    mmcif: str = None

    def __post_init__(self):
        if self.sequence is None and self.pdb is None and self.mmcif is None:
            raise TypeError("At least one of the following parameters must be specified: 'sequence', 'pdb' or 'mmcif'")
