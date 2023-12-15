"""
Functions to handle the input to the modules. 
"""
import os
import json
from pathlib import Path
import logging
import yaml
import configargparse
from dask.distributed import Client, LocalCluster
import dask.config


def configure(args: list, module_name: str = None) -> dict:
    """
    Process the given arguments, identify the configuration filepath, and extract the correct configuration settings for
    the requested module.

    Parameters
    ----------
    args : list
        List of input arguments
    module_name : str, optional
        Module name. If given, will return the specific configuration for the requested module. Otherwise return all
        configurations.

    Returns
    -------
    config : dict
        Dictionary with user configurations

    """

    # TODO : allow input of default arguments
    # Create an instance of arg_parser and define all the user input arguments
    arg_parser = configargparse.ArgumentParser()
    arg_parser.add_argument(
        "configfile_path", type=str, help="The full path to the .json/.yaml config file"
    )

    # deal with configuration
    if args is not None:
        commandline_args = arg_parser.parse_known_args(args=args)[0]
    else:
        raise SystemError("Need to provide a path to a configuration file")

    if not os.path.exists(commandline_args.configfile_path):
        raise SystemError(
            f"Configuration file {commandline_args.configfile_path} does not exist"
        )

    config = load_config(commandline_args.configfile_path)

    # TODO: override config with command line parameters
    if module_name is not None:
        if "modules" not in config:
            raise SystemError(
                f"No modules section found in configuration file."
                f"Cannot proceed to configure {module_name}"
            )
        if module_name not in config["modules"]:
            raise SystemError(
                f"Module name {module_name} not found in configuration file"
            )
        config = config["modules"][module_name]

    return config


def load_config(path: str) -> dict:
    """
    Reads configuration from configuration file, which may be in .yml or .json format.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    config : dict
        Dictionary with configuration settings
    """

    local_path = Path(path)
    with open(local_path, "r") as stream:
        ext = local_path.suffix
        if ext in [".yml", ".yaml"]:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        elif ext in [".json"]:
            config = json.load(stream)
        else:
            raise SystemError(f"Configuration file extension {ext} unknown")
    return config


def preamble(args, module_name):
    """
    Generic preamble for all modules.
    Read configuration file, set up dask client and logging.

    Parameters
    ----------
    args : list
        List of command line arguments
    module_name : str

    Returns
    -------
    config : dict
        Dictionary with user configurations
    client : dask.distributed.Client
        Dask client
    """

    config = configure(args[1:], module_name)

    dask.config.set(
        {
            "distributed.scheduler.worker-ttl": None,
            "logging.distributed": "error",
        }
    )

    cluster = LocalCluster(
        n_workers=config.get("n_workers", None),
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.captureWarnings(True)

    return config, client
