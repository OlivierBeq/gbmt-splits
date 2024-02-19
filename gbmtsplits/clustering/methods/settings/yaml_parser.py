# -*- coding: utf-8 -*-

# The code below originates from DataSAIL: https://github.com/kalininalab/DataSAIL

# MIT License
#
# Copyright (c) 2023 Roman Joeres
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
from pathlib import Path
from pydoc import locate
from typing import Optional, List, Sequence, Literal

import yaml

from . import YAML_FILE_NAMES


class MultiYAMLParser(argparse.ArgumentParser):
    def __init__(self, algo_name):
        """
        Initialize the argument parser for DataSAIL. This is a wrapper around the standard argparse.ArgumentParser.

        Args:
            algo_name: Name of the algorithm to parse arguments for.
        """
        super().__init__()
        self.fos_map = {}
        if algo_name is not None:
            self.add_yaml_arguments(YAML_FILE_NAMES[algo_name])

    def parse_args(self, args: Optional[Sequence[str]] = ...) -> argparse.Namespace:
        """
        Parse the arguments provided by the user. This prepends some preprocessing to the arguments before sending them
        to the actual parsing.

        Args:
            args: Arguments provided by the user.

        Returns:
            Namespace of the parsed arguments.
        """
        # args = args.split(" ") if " " in args else (args if isinstance(args, list) else [args])
        if isinstance(args, str):
            if " " in args:
                args = args.split(" ")
            elif len(args) > 0:
                args = [args]
        return super().parse_args(args)

    def add_yaml_arguments(self, yaml_filepath: Path) -> None:
        """
        Add arguments to the parser based on a YAML file.

        Args:
            yaml_filepath: Path to the YAML file to read the arguments from.
        """
        with open(Path(__file__).parent.resolve() / yaml_filepath, "r") as data:
            data = yaml.safe_load(data)
        for name, values in data.items():
            kwargs = {"dest": name.replace("-", "_"), "type": locate(values["type"])}
            if kwargs["type"] == bool:
                if not values["default"]:
                    kwargs.update({"action": "store_true", "default": False})
                else:
                    kwargs.update({"action": "store_false", "default": True})
                del kwargs["type"]
            else:
                if values["cardinality"] != 0:
                    kwargs["nargs"] = values["cardinality"]
                if values["default"] is not None:
                    kwargs["default"] = values["default"]
            self.fos_map[name.replace("-", "_")] = values.get("fos", 0)
            super().add_argument(
                *values["calls"],
                **kwargs,
            )

    def get_user_arguments(self, args: argparse.Namespace, ds_args: List[str], fos: Literal[0, 1] = 0) -> str:
        """
        Get the arguments that the user provided to the program that differ from default values.

        Args:
            args: Arguments provided by the user.
            ds_args: Arguments that are optimized by DataSAIL and extracted differently.
            fos: Group of arguments to be considered (used for cd-hit submodules); one of {0, 1}.

        Returns:
            String representation of the arguments that the user provided for the program to be passed to subprograms.
        """
        cleaned_args = namespace_diff(args, self.parse_args([]))  # the non-standard arguments
        action_map = {action.dest: action.option_strings[0] for action in self._actions}
        fos = {fos, 2}

        for key in ds_args:
            if key in cleaned_args:
                del cleaned_args[key]

        return " ".join([f"{action_map[key]} {value}" for key, value in cleaned_args.items() if self.fos_map[key] in fos])


def namespace_diff(a: argparse.Namespace, b: argparse.Namespace) -> dict:
    """
    Get the difference between two namespaces.

    Args:
        a: First namespace to compare.
        b: Second namespace to compare.

    Returns:
        Dictionary of all attributes that are different between the two namespaces.
    """
    output = {}
    if a is None:
        return output
    for key, value in vars(a).items():
        if not hasattr(b, key) or getattr(b, key) != value:
            output[key] = value
    return output


def kwargs_to_namespace(**kwargs) -> argparse.Namespace:
    """
    Create a Namespace from keyword arguments
    Parameters
    ----------
    kwargs :
        keyword arguments.
    Returns
    -------
        a Namespace with the given arguments set.
    """
    ns = argparse.Namespace()
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns
