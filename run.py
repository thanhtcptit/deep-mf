# -*- coding: utf-8 -*-
from email.policy import default
import os
import sys

import pkgutil
import importlib


def import_submodules(package_name: str) -> None:
    importlib.invalidate_caches()
    sys.path.append(".")

    module = importlib.import_module(package_name)
    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    for module_finder, name, _ in pkgutil.walk_packages(path):
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)


import_submodules("src")
from src.utils.params import Params
from src.utils.args_parser import Subcommand, ArgumentParserWithDefaults


def main(subcommand_overrides={}):
    parser = ArgumentParserWithDefaults()

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands = {
        "preprocess": Preprocess(),
        "train": Train(),
        "hyp_train": HyperparamsSearchTraining(),
        "hyp_recon": HyperparamsSearchReconstruction(),
        "eval": Evaluate(),
        "eval_kw": EvaluateKW(),
        "vis_sim": VisualizeSimilarItems(),
        "export": ExportModel(),
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        if name != "configure":
            subparser.add_argument("--include-package",
                                   type=str,
                                   action="append",
                                   default=[],
                                   help="additional packages to include")

    args = parser.parse_args()
    if "func" in dir(args):
        for package_name in getattr(args, "include_package", ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


class Preprocess(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run data preprocess"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to the json config file")
        subparser.add_argument(
            "-f", "--force", action="store_true",
            help="force override serialization dir")

        subparser.set_defaults(func=preprocess)
        return subparser


def preprocess(args):
    from src.data.preprocess import main as func
    return func(args.config_path, args.force)


class Train(Subcommand):
    def add_subparser(self, name, parser):
        description = """Train the specified model on the specified dataset"""
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to parameter file describing the model to be trained")
        subparser.add_argument(
            "dataset_path", type=str,
            help="path to the training dataset")
        subparser.add_argument(
            "-s", "--save_dir", type=str, default="",
            help="directory in which to save the model and its logs")
        subparser.add_argument(
            "-r", "--recover", action="store_true",
            help="recover training from the state in serialization_dir")
        subparser.add_argument(
            "-f", "--force", action="store_true",
            help="force override serialization dir")

        subparser.set_defaults(func=train_model)

        return subparser


def train_model(args):
    from src.train import train as func
    return func(args.config_path, args.dataset_path, args.save_dir, args.recover, args.force)


class HyperparamsSearchTraining(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run hyperparams search for training"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to the json config file")
        subparser.add_argument(
            "dataset_path", type=str,
            help="path to the training dataset")
        subparser.add_argument(
            "-t", "--test_dataset_path", type=str, default=None,
            help="path to evaluate dataset")
        subparser.add_argument(
            "-a", "--additional_dataset_path", type=str, default=None,
            help="path to additional dataset")
        subparser.add_argument(
            "-n", "--num_trials", type=int, default=100,
            help="number of trials to run")
        subparser.add_argument(
            "-f", "--force", action="store_true",
            help="force override serialization dir")

        subparser.set_defaults(func=hyperparams_search_training)
        return subparser


def hyperparams_search_training(args):
    from src.train import hyperparams_search_training as func
    return func(args.config_path, args.dataset_path, args.test_dataset_path, args.additional_dataset_path,
                args.num_trials, args.force)


class HyperparamsSearchReconstruction(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run hyperparams search for reconstruction"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to the json config file")
        subparser.add_argument(
            "checkpoint_dir", type=str,
            help="path to the trained model checkpoint")
        subparser.add_argument(
            "test_dataset_path", type=str,
            help="path to the test dataset")
        subparser.add_argument(
            "additional_dataset_path", type=str,
            help="path to additional dataset")
        subparser.add_argument(
            "-n", "--num_trials", type=int, default=100,
            help="number of trials to run")

        subparser.set_defaults(func=hyperparams_search_training)
        return subparser


def hyperparams_search_training(args):
    from src.train import hyperparams_search_reconstruction as func
    return func(args.config_path, args.checkpoint_dir, args.test_dataset_path, args.additional_dataset_path,
                args.num_trials)


class Evaluate(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run evaluation"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "checkpoint_path", type=str,
            help=("path to the model checkpoint"))
        subparser.add_argument(
            "-d", "--test_dataset_path", type=str, default=None,
            help="path to evaluate dataset")
        subparser.set_defaults(func=evaluate)
        return subparser


def evaluate(args):
    from src.train import test as func
    return func(args.checkpoint_path, args.test_dataset_path)


class EvaluateKW(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run evaluation on kw"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "checkpoint_path", type=str,
            help=("path to the model checkpoint"))
        subparser.add_argument(
            "-d", "--test_dataset_path", type=str, default=None,
            help="path to evaluate dataset")
        subparser.add_argument(
            "-a", "--additional_dataset_path", type=str, default=None,
            help="path to additional dataset")
        subparser.add_argument(
            "-c", "--reconstruction_config", type=str, default=None,
            help="path to the reconstruction config")
        subparser.set_defaults(func=evaluate_kw)
        return subparser


def evaluate_kw(args):
    from src.train import test_keyword as func
    return func(args.checkpoint_path, args.test_dataset_path, args.additional_dataset_path,
                args.reconstruction_config)


class VisualizeSimilarItems(Subcommand):
    def add_subparser(self, name, parser):
        description = """Visualize similar item embeddings"""
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            "checkpoint_path", type=str, help="path to model checkpoint")
        subparser.add_argument(
            "-f", "--force", action="store_true", help="override the ann tree")
        subparser.set_defaults(func=visualize_similar_items)

        return subparser


def visualize_similar_items(args):
    from src.visualization.visualize_embeddings import get_similar_items as func
    return func(args.checkpoint_path, args.force)


class ExportModel(Subcommand):
    def add_subparser(self, name, parser):
        description = """Export model for serving"""
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            "checkpoint_path", type=str, help="path to model checkpoint")
        subparser.add_argument(
            "-o", "--output_dir", type=str, help="path to output dir")
        subparser.set_defaults(func=export_model)

        return subparser


def export_model(args):
    raise NotImplementedError()


def run():
    main()


if __name__ == "__main__":
    run()
