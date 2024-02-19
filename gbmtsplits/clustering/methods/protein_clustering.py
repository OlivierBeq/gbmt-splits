# -*- coding: utf-8 -*-

import importlib.metadata
import os
import shutil
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from ...clustering.interface import ProteinSequenceClusteringMethod, ProteinCrystalStructureClusteringMethod
from .settings.yaml_parser import MultiYAMLParser, kwargs_to_namespace
from . import io, utils, grakelio
from .. import Protein


class MMseqs2(ProteinSequenceClusteringMethod):
    def __init__(self, threads: int = 1, log_dir: Path = None, **kwargs):
        """Instantiate a MMseqs2 clustering tool for protein sequences.

        MMseqs2 does not output pairwise similarities, therefore, one parameter
        must be adjusted to determine how fine or coarse the clustering will be to
        find the best clustering. This is -c and it is automatically adjusted
        and searched to find a good clustering to start splitting the data.

        Parameters
        ----------
        threads : int
            number of concurrent threads
        log_dir : Path
            path to the folder in which log files will be created
        kwargs : dict[str, str]
            keyword arguments to be passed to the CD-Hit executable
        """
        if not self.is_installed():
            raise RuntimeError('MMseqs2 is not installed')

        super().__init__()
        kwargs = kwargs_to_namespace(**kwargs)
        parser = MultiYAMLParser('MMSEQS2')
        self.user_args = parser.get_user_arguments(kwargs, ["c"])
        other_args = parser.parse_args([])
        self.optim_vals = (other_args.c,)  # values to be optimized
        self.threads = threads
        self.log_dir = log_dir

    def is_installed(self):
        return shutil.which("mmseqs") is not None

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein sequences using MMseqs2.

        Parameters
        ----------
        string_list : list[Protein]
            List of proteins to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        results_folder = Path("mmseqs_results")
        if results_folder.exists():
            shutil.rmtree(results_folder.as_posix(), ignore_errors=True)
        results_folder.mkdir()
        fasta_path = io.sequences_to_fasta(string_list, results_folder)
        result = utils.cluster_param_binary_search(results_folder=results_folder,
                                                   fasta_file=fasta_path,
                                                   sequence_ids=[],
                                                   init_args=self.optim_vals,
                                                   min_args=(0.1,),
                                                   max_args=(1,),
                                                   user_args=self.user_args,
                                                   threads=self.threads,
                                                   trial=self.trial,
                                                   args2str=lambda x: f"-c {x[0]}",
                                                   gen_args=lambda x, y: ((x[0] + y[0]) / 2,),
                                                   log_dir=self.log_dir
                                                   )
        return result

    def trial(self,
              results_folder: Path,
              fasta_file: Path,
              tune_args: str,
              user_args: str,
              threads: int = 1,
              log_file: Path = None
              ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run MMseqs2 on the dataset with the given sequence similarity defined by add_args.

        Args:
            results_folder: folder containing input and output files
            fasta_file: FASTA file containing protein sequences
            tune_args: Generated command line for arguments being tuned
            user_args: Additional arguments specifying the sequence similarity parameter
            threads: Number of threads to use for one MMseqs2 run
            log_file: Filepath to log the output to

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """

        cmd = (f"cd {results_folder} && "
               f"mmseqs easy-cluster {fasta_file} mmseqs_out mmseqs_tmp --threads {threads} "
               f"{tune_args} {user_args} ")

        if log_file is None:
            cmd += "> /dev/null 2>&1"
        else:
            cmd += f"> {log_file.resolve()}"

        os.system(cmd)

        if not (results_folder / "mmseqs_out_cluster.tsv").is_file():
            raise ValueError("Something went wrong with MMseqs2. The output file does not exist.")

        cluster_map = self.get_mmseqs2_map(results_folder / "mmseqs_out_cluster.tsv")
        cluster_names = list(set(cluster_map.values()))
        cluster_sim = np.ones((len(cluster_names), len(cluster_names)))

        shutil.rmtree(results_folder, ignore_errors=True)

        return cluster_names, cluster_map, cluster_sim

    def get_mmseqs2_map(self, cluster_file: Path) -> dict[str, str]:
        """
        Read clusters from mmseqs output into map from cluster members to cluster representatives (cluster names).

        Args:
            cluster_file (str): Filepath of file containing the mapping information

        Returns:
            Map from cluster--members to cluster-representatives (cluster-names)
        """
        mapping = {}
        with open(cluster_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0 and "\t" not in line:
                    return self.get_mmseqs_map_old(cluster_file)

                words = line.strip().split('\t')
                if len(words) != 2:
                    continue
                cluster_head, cluster_member = words
                mapping[cluster_member] = cluster_head
        return mapping

    def get_mmseqs_map_old(self, cluster_file: Path) -> dict[str, str]:
        """
        This is a helper method for get_mmseqs_map that is necessary when running on Windows and in a Python3.8 build.
        In this case, MMseqs struggles with the different linebreaks of Linux and Windows.

        Args:
            cluster_file (str): Filepath of file containing the mapping information

        Returns:
            Map from cluster-members to cluster-representatives (cluster-names)
        """
        mapping = {}
        rep = ""
        # The file basically contains \n\t-separated values
        with open(cluster_file, "r") as f:
            for line in f.readlines():
                if rep == "":
                    rep = line.strip()
                else:
                    mapping[rep] = line.strip()
                    rep = ""
        return mapping


class CDHit(ProteinSequenceClusteringMethod):
    def __init__(self, threads: int = 1, log_dir: Path = None, **kwargs):
        """Instantiate a CD-HIT clustering tool for protein sequences.

        CD-HIT does not output pairwise similarities, therefore, two parameters
        must be adjusted to determine how fine or coarse the clustering will be to
        find the best clustering. Those are -n and -c. Those are automatically adjusted
        and searched to find a good clustering to start splitting the data.

        Parameters
        ----------
        threads : int
            number of concurrent threads
        log_dir : Path
            path to the folder in which log files will be created
        kwargs : dict[str, str]
            keyword arguments to be passed to the CD-Hit executable
        """
        if not self.is_installed():
            raise RuntimeError('CH-HIT is not installed')

        super().__init__()
        kwargs = kwargs_to_namespace(**kwargs)
        parser = MultiYAMLParser('CDHIT')
        self.user_args = parser.get_user_arguments(kwargs, ["c", "n"])
        other_args = parser.parse_args([])
        self.optim_vals = (other_args.c, other_args.n)  # values to be optimized
        self.threads = threads
        self.log_dir = log_dir

    def is_installed(self):
        return shutil.which("cd-hit") is not None

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein sequences using CD-HIT.

        Parameters
        ----------
        string_list : list[Protein]
            List of proteins to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        results_folder = Path("cdhit_results")
        if results_folder.exists():
            shutil.rmtree(results_folder.as_posix(), ignore_errors=True)
        results_folder.mkdir()
        fasta_path = io.sequences_to_fasta(string_list, results_folder)
        result = utils.cluster_param_binary_search(results_folder=results_folder,
                                                   fasta_file=fasta_path,
                                                   sequence_ids=[],
                                                   init_args=self.optim_vals,
                                                   min_args=(0.4, 2),
                                                   max_args=(1, 5),
                                                   user_args=self.user_args,
                                                   threads=self.threads,
                                                   trial=self.trial,
                                                   args2str=lambda x: f"-c {x[0]} -n {x[1]} -l {x[1] - 1}",
                                                   gen_args=lambda x, y: ((x[0] + y[0]) / 2, self.c2n((x[0] + y[0]) / 2)),
                                                   log_dir=self.log_dir
                                                   )
        return result

    def trial(self,
              results_folder: Path,
              fasta_file: Path,
              tune_args: str,
              user_args: str,
              threads: int = 1,
              log_file: Path = None
              ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run CD-HIT on the dataset with the given sequence similarity defined by add_args.

        Args:
            results_folder: folder containing input and output files
            fasta_file: FASTA file containing protein sequences.
            tune_args: Generated command line for arguments being tuned.
            user_args: Additional arguments specifying the sequence similarity parameter.
            threads: Number of threads to use for one CD-HIT run.
            log_file: Filepath to log the output to.

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """

        cmd = (f"cd {results_folder} && "
               f"cd-hit -i {Path('..') / fasta_file} -o clusters -d 0 -T {threads} "
               f"{tune_args} {user_args} ")

        if log_file is None:
            cmd += "> /dev/null 2>&1"
        else:
            cmd += f"> {log_file.resolve()}"

        os.system(cmd)

        if not (results_folder / "clusters.clstr").is_file():
            raise ValueError("Something went wrong with cd-hit. The output file does not exist.")

        cluster_map = self.get_cdhit_map(results_folder / "clusters.clstr")
        cluster_names = list(set(cluster_map.values()))
        cluster_sim = np.ones((len(cluster_names), len(cluster_names)))

        shutil.rmtree(results_folder, ignore_errors=True)

        return cluster_names, cluster_map, cluster_sim

    def get_cdhit_map(self, cluster_file: Path) -> dict[str, str]:
        """
        Read the cluster assignment from the output of CD-HIT.

        Args:
            cluster_file (str): filepath of the file that stores the cluster assignment.

        Returns:
            Map from cluster-members to cluster-representatives (cluster-names)
        """
        mapping = {}
        rep = ""
        members = []
        with open(cluster_file, "r") as data:
            for line in data.readlines():
                line = line.strip()
                if line[0] == ">":
                    if rep != "":
                        mapping[rep] = rep
                        for name in members:
                            mapping[name] = rep
                    rep = ""
                    members = []
                elif line[-1] == "*":
                    rep = line.split(">")[1].split("...")[0]
                else:
                    members.append(line.split(">")[1].split("...")[0])
        mapping[rep] = rep
        for name in members:
            mapping[name] = rep
        return mapping

    def c2n(self, c: float):
        """
        For an input value for the C-parameter to CD-HIT, return an appropriate value for the parameter n.

        Args:
            c: c parameter to CD-HIT

        Returns:
            An according value for n based on c
        """
        if 0.4 <= c < 0.5:
            return 2
        elif 0.5 <= c < 0.6:
            return 3
        elif 0.6 <= c < 0.7:
            return 4
        else:
            return 5


class Mash(ProteinSequenceClusteringMethod):
    def __init__(self, threads: int = 1, log_dir: Path = None, **kwargs):
        """Instantiate a MASH clustering tool for protein sequences.

        MASH produces pairwise distances.

        Parameters
        ----------
        threads : int
            number of concurrent threads
        log_dir : Path
            path to the folder in which log files will be created
        kwargs : dict[str, str]
            keyword arguments to be passed to the MMseqs2 executable
        """
        if not self.is_installed():
            raise RuntimeError('MASH is not installed')

        super().__init__()
        kwargs = kwargs_to_namespace(**kwargs)
        parser = MultiYAMLParser('MASH')
        self.sketch_args = parser.get_user_arguments(kwargs, [], 0)
        self.dist_args = parser.get_user_arguments(kwargs, [], 1)
        other_args = parser.parse_args([])
        self.optim_vals = (other_args.c,)  # values to be optimized
        self.threads = threads
        self.log_dir = log_dir

    def is_installed(self):
        return shutil.which("mash") is not None

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein sequences using MASH.

        Parameters
        ----------
        string_list : list[Protein]
            List of proteins to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        results_folder = Path("mash_results")
        if results_folder.exists():
            shutil.rmtree(results_folder.as_posix(), ignore_errors=True)
        results_folder.mkdir()
        fasta_path = io.sequences_to_fasta(string_list, results_folder)
        result = self.run(results_folder=results_folder,
                          fasta_file=fasta_path,
                          sequence_ids=[seq.id for seq in string_list],
                          threads=self.threads,
                          log_file= (self.log_dir / 'mash.log') if self.log_dir is not None else None
                          )
        return result

    def run(self,
            results_folder: Path,
            fasta_file: Path,
            sequence_ids: list[str],
            threads: int = 1,
            log_file: Path = None
    ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run MASH on the dataset with the given sequence similarity defined by add_args.

        Args:
            results_folder: folder containing input and output files
            fasta_file: FASTA file containing protein sequences.
            sequence_ids: List of sequence IDs.
            threads: Number of threads to use for one MASH run.
            log_file: Filepath to log the output to.

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """


        cmd = (f"cd {results_folder} && "
               f"mash sketch -p {threads} -o ./cluster {Path('..') / fasta_file} {self.sketch_args} && "
               f"mash dist -p {threads} {self.dist_args} -t cluster.msh cluster.msh > cluster.tsv")

        if log_file is None:
            cmd += "> /dev/null 2>&1"
        else:
            cmd += f"> {log_file.resolve()}"

        os.system(cmd)

        if not (results_folder / "cluster.tsv").is_file():
            raise ValueError("Something went wrong with MASH. The output file does not exist.")

        cluster_names = sequence_ids
        cluster_map = dict((n, n) for n in cluster_names)
        cluster_dist = self.read_mash_tsv(results_folder / "cluster.tsv", len(cluster_names))

        shutil.rmtree(results_folder, ignore_errors=True)

        return cluster_names, cluster_map, cluster_dist

    def read_mash_tsv(self, cluster_file: Path, num_entities: int) -> np.ndarray:
        """
        Read in the TSV file with pairwise distances produced by MASH.

        Args:
            filename: Filename of the file to read from
            num_entities: Number of entities in the set

        Returns:
            Symmetric 2D-numpy array storing pairwise distances
        """
        output = np.zeros((num_entities, num_entities))
        with open(cluster_file, "r") as data:
            for i, line in enumerate(data.readlines()[1:]):
                for j, val in enumerate(line.strip().split("\t")[1:]):
                    output[i, j] = float(val)
        return output


class MMseqs2_spectral_clustering(ProteinSequenceClusteringMethod):
    def __init__(self, threads: int = 1, log_dir: Path = None, **kwargs):
        """Instantiate a MMseqs2 clustering tool for protein sequences with additional spectral clustering.

        It obtains pairwise similarities between protein sequences and uses them for spectral clustering.
        This is more accurate than using MMseqs2 to cluster sequences directly, at the expense of speed.

        Parameters
        ----------
        threads : int
            number of concurrent threads
        log_dir : Path
            path to the folder in which log files will be created
        kwargs : dict[str, str]
            keyword arguments to be passed to the MMseqs2 executable
        """
        if not self.is_installed():
            raise RuntimeError('MMseqs2 is not installed')

        super().__init__()
        kwargs = kwargs_to_namespace(**kwargs)
        parser = MultiYAMLParser('MMSEQS_SPECTCLUST')
        self.prefilter_args = parser.get_user_arguments(kwargs, [], 0)
        self.align_args = parser.get_user_arguments(kwargs, [], 1)
        other_args = parser.parse_args([])
        self.threads = threads
        self.log_dir = log_dir

    def is_installed(self):
        return shutil.which("mmseqs") is not None

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein sequences using MMseqs2.

        Parameters
        ----------
        string_list : list[Protein]
            List of proteins to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        results_folder = Path("mmseqs2_spectclust_results")
        if results_folder.exists():
            shutil.rmtree(results_folder.as_posix(), ignore_errors=True)
        results_folder.mkdir()
        fasta_path = io.sequences_to_fasta(string_list, results_folder)
        result = self.run(results_folder=results_folder,
                          fasta_file=fasta_path,
                          sequence_ids=[seq.id for seq in string_list],
                          threads=self.threads,
                          log_file= (self.log_dir / 'mash.log') if self.log_dir is not None else None
                          )
        return result

    def run(self,
            results_folder: Path,
            fasta_file: Path,
            sequence_ids: list[str],
            threads: int = 1,
            log_file: Path = None
            ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run MMseqs2 on the dataset with additional spectral clustering.

        Args:
            results_folder: folder containing input and output files
            fasta_file: FASTA file containing protein sequences.
            sequence_ids: List of sequence IDs.
            threads: Number of threads to use for one MASH run.
            log_file: Filepath to log the output to.

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """

        cmd = lambda x: (f"mkdir {results_folder} && "
                         f"cd {results_folder} && "
                         f"mmseqs createdb {Path('..') / fasta_file} seqs.db {x} && "
                         f"mmseqs prefilter seqs.db seqs.db seqs.pref --threads {threads} {self.prefilter_args} {x} && "
                         f"mmseqs align seqs.db seqs.db seqs.pref seqs.ali -e inf --threads {threads} {self.align_args} {x} && "
                         f"mmseqs convertalis seqs.db seqs.db seqs.ali alis.tsv --format-mode 4 --format-output query,target,fident --threads {threads} {x}")

        if log_file is None:
            cmd = cmd("> /dev/null 2>&1")
        else:
            cmd = cmd(f">> {log_file.resolve()}")

        os.system(cmd)

        if not (results_folder / "alis.tsv").is_file():
            raise ValueError("Something went wrong with MMseqs2. The output file does not exist.")

        df = pd.read_csv(results_folder / "alis.tsv", sep="\t")
        table = df.pivot(index="query", columns="target", values="fident").fillna(0).to_numpy()

        shutil.rmtree(results_folder, ignore_errors=True)

        return sequence_ids, {n: n for n in sequence_ids}, table


class TMAlign(ProteinCrystalStructureClusteringMethod):
    def __init__(self, log_dir: Path = None):
        """Instantiate a TM-align clustering tool for protein crystal structures.

        It first generates pairwise optimized residue-to-residue alignment based on structural similarity
        using heuristic dynamic programming iterations. Then, pairwise optimal superpositions of the
        structures are built on the detected alignment. Finally, the structural similarity is scaled with TM-score.

        Parameters
        ----------
        kwargs : dict[str, str]
            keyword arguments to be passed to the MMseqs2 executable
        """
        if not self.is_installed():
            raise RuntimeError('TM-align is not installed')

        super().__init__()
        self.log_dir = log_dir

    def is_installed(self):
        return shutil.which("TMalign") is not None

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein crystal structures using TM-align.

        Parameters
        ----------
        string_list : list[Protein]
            List of protein crystal structures to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        return self.run(string_list)

    def run(self,
            protein_structures: list[Protein],
            ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run TM-align on the dataset.

        Args:
            protein_structures: protein crystal structures to cluster.

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """

        results_folder = Path("tmalign_results")
        if results_folder.exists():
            shutil.rmtree(results_folder.as_posix(), ignore_errors=True)
        results_folder.mkdir()

        cmd = (f"mkdir {results_folder} && "
               f"cd {results_folder}")

        # Iterating pairs of structures as TM-align works only on pairs
        for s1, s2 in combinations(protein_structures, 2):
            file1 = io.structure_to_pdb(s1, results_folder)
            file2 = io.structure_to_pdb(s2, results_folder)
            cmd += (f" && TMalign {file1} {file2} > out_{s1.id}_{s2.id}.txt ")


        os.system(cmd)

        cluster_names = [struct.id for struct in protein_structures]
        cluster_map = dict((n, n) for n in cluster_names)
        cluster_sim = self.read_tmalign_folder(cluster_names, results_folder)

        shutil.rmtree(results_folder, ignore_errors=True)

        return cluster_names, cluster_map, cluster_sim

    def read_tmalign_folder(self, structure_ids: list[str], tmalign_folder: Path) -> np.ndarray:
        """
        Read clusters from TM-align output into map from cluster members to cluster representatives (cluster names).

        Args:
            dataset: Dataset with the data to cluster
            tmalign_folder: Path to the folder of file containing the mapping information

        Returns:
            Map from cluster-members to cluster-representatives (cluster-names)
        """
        sims = np.ones((len(structure_ids), len(structure_ids)))
        for i, name1 in enumerate(structure_ids):
            for j, name2 in enumerate(structure_ids[i + 1:]):
                sims[i, i + j + 1] = self.read_tmalign_file(tmalign_folder / f"out_{name1}_{name2}.txt")
                sims[i, i + j + 1] = sims[i + j + 1, i]
        return sims

    def read_tmalign_file(self, filepath: Path) -> float:
        """
        Read one TM-align file holding the output of one tmalign run.

        Args:
            filepath: path to the file to read from

        Returns:
            The average tm-score of both directions of that pairwise alignment
        """
        with open(filepath, "r") as data:
            return sum(map(lambda x: float(x.split(" ")[1]), data.readlines()[17:19])) / 2


class FoldSeek(ProteinCrystalStructureClusteringMethod):
    def __init__(self, threads: int = 1, log_dir: Path = None, **kwargs):
        """Instantiate a FoldSeek clustering tool for protein crystal structures.

        FoldSeek first produces a pairwise similarity matrix and then clusters crystal structures.

        Parameters
        ----------
        threads : int
            number of concurrent threads
        log_dir : Path
            path to the folder in which log files will be created
        kwargs : dict[str, str]
            keyword arguments to be passed to the FoldSeek executable
        """
        if not self.is_installed():
            raise RuntimeError('FoldSeek is not installed')

        super().__init__()
        kwargs = kwargs_to_namespace(**kwargs)
        parser = MultiYAMLParser('FOLDSEEK')
        self.user_args = parser.get_user_arguments(kwargs, [])
        self.threads = threads
        self.log_dir = log_dir

    def is_installed(self):
        return shutil.which("foldseek") is not None

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein crystal structures using TM-align.

        Parameters
        ----------
        string_list : list[Protein]
            List of protein crystal structures to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        return self.run(string_list)

    def run(self,
            protein_structures: list[Protein],
            ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run FoldSeek on the dataset.

        Args:
            protein_structures: protein crystal structures to cluster.

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """
        results_folder = Path("foldseek_results")
        struct_folder = results_folder / 'structures'
        if results_folder.exists():
            shutil.rmtree(results_folder.as_posix(), ignore_errors=True)
        results_folder.mkdir()


        for struct in protein_structures:
            io.structure_to_pdb(struct, results_folder)

        cmd = (f"mkdir {struct_folder} && "
               f"cd {results_folder} && "
               f"foldseek "
               f"easy-search "
               f"{struct_folder} "
               f"{struct_folder} "
               f"aln.m8 "
               f"tmp "
               f"--format-output 'query,target,fident' "
               f"-e inf "
               f"--threads {self.threads} "
               f"{self.user_args}")

        if self.log_dir is None:
            cmd += "> /dev/null 2>&1"
        else:
            cmd += f"> {(self.log_dir / 'foldseek.log').resolve()}"

        if struct_folder.exists():
            cmd = f"rm -rf {struct_folder} && " + cmd

        os.system(cmd)

        if not (results_folder / "aln.m8").exists():
            raise ValueError("Something went wrong with FoldSeek. The output file does not exist.")

        namap = dict((struct.id, i) for i, struct in enumerate(protein_structures))
        cluster_sim = np.zeros((len(protein_structures), len(protein_structures)))
        with open(f"{results_folder}/aln.m8", "r") as data:
            for line in data.readlines():
                q1, q2, sim = line.strip().split("\t")[:3]
                if "_" in q1 and "." in q1 and q1.rindex("_") > q1.index("."):
                    q1 = "_".join(q1.split("_")[:-1])
                if "_" in q2 and "." in q2 and q2.rindex("_") > q2.index("."):
                    q2 = "_".join(q2.split("_")[:-1])
                q1 = q1.replace(".pdb", "")
                q2 = q2.replace(".pdb", "")
                cluster_sim[namap[q1], namap[q2]] = sim
                cluster_sim[namap[q2], namap[q1]] = sim

        cluster_names = [struct.id for struct in protein_structures]
        cluster_map = dict((n, n) for n in cluster_names)

        shutil.rmtree(results_folder, ignore_errors=True)

        return cluster_names, cluster_map, cluster_sim


class WLK(ProteinCrystalStructureClusteringMethod):
    def __init__(self, threads: int = 1, log_dir: Path = None, **kwargs):
        """Instantiate a Weisfeiler-Lehman kernel-based clustering.

        FoldSeek first produces a pairwise similarity matrix and then clusters crystal structures.

        Parameters
        ----------
        threads : int
            number of concurrent threads
        log_dir : Path
            path to the folder in which log files will be created
        kwargs : dict[str, str]
            keyword arguments to be passed to the FoldSeek executable
        """
        #TODO: change from here on
        if not self.is_installed():
            raise RuntimeError('FoldSeek is not installed')

        super().__init__()
        kwargs = kwargs_to_namespace(**kwargs)
        parser = MultiYAMLParser('FOLDSEEK')
        self.user_args = parser.get_user_arguments(kwargs, [])
        self.threads = threads
        self.log_dir = log_dir

    def is_installed(self):
        try:
            importlib.metadata.version('grakel')
        except importlib.metadata.PackageNotFoundError:
            return False
        else:
            return True

    def __call__(self, string_list: list[Protein]) -> tuple[list[str], dict[str, str], np.ndarray]:
        """Cluster protein crystal structures using TM-align.

        Parameters
        ----------
        string_list : list[Protein]
            List of protein crystal structures to cluster.

        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices
            and values are the indices of proteins sequences.
        """
        return self.run(string_list)

    def run(self,
            protein_structures: list[Protein],
            ) -> tuple[list[str], dict[str, str], np.ndarray]:
        """
        Run FoldSeek on the dataset.

        Args:
            protein_structures: protein crystal structures to cluster.

        Returns:
            A tuple containing
              - the names of the clusters (cluster representatives)
              - the mapping from cluster members to the cluster names (cluster representatives)
              - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
        """

        graphs = [grakel.pdb_to_grakel(struct) for struct in protein_structures]

        cmd = (f"mkdir {struct_folder} && "
               f"cd {results_folder} && "
               f"foldseek "
               f"easy-search "
               f"{struct_folder} "
               f"{struct_folder} "
               f"aln.m8 "
               f"tmp "
               f"--format-output 'query,target,fident' "
               f"-e inf "
               f"--threads {self.threads} "
               f"{self.user_args}")

        if self.log_dir is None:
            cmd += "> /dev/null 2>&1"
        else:
            cmd += f"> {(self.log_dir / 'foldseek.log').resolve()}"

        if struct_folder.exists():
            cmd = f"rm -rf {struct_folder} && " + cmd

        os.system(cmd)

        if not (results_folder / "aln.m8").exists():
            raise ValueError("Something went wrong with FoldSeek. The output file does not exist.")

        namap = dict((struct.id, i) for i, struct in enumerate(protein_structures))
        cluster_sim = np.zeros((len(protein_structures), len(protein_structures)))
        with open(f"{results_folder}/aln.m8", "r") as data:
            for line in data.readlines():
                q1, q2, sim = line.strip().split("\t")[:3]
                if "_" in q1 and "." in q1 and q1.rindex("_") > q1.index("."):
                    q1 = "_".join(q1.split("_")[:-1])
                if "_" in q2 and "." in q2 and q2.rindex("_") > q2.index("."):
                    q2 = "_".join(q2.split("_")[:-1])
                q1 = q1.replace(".pdb", "")
                q2 = q2.replace(".pdb", "")
                cluster_sim[namap[q1], namap[q2]] = sim
                cluster_sim[namap[q2], namap[q1]] = sim

        cluster_names = [struct.id for struct in protein_structures]
        cluster_map = dict((n, n) for n in cluster_names)

        shutil.rmtree(results_folder, ignore_errors=True)

        return cluster_names, cluster_map, cluster_sim
