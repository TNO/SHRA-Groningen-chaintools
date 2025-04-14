"""
Functions to handle the input to the modules.
"""

import os
import json
from copy import deepcopy
from pathlib import Path
import collections
from itertools import islice
import logging
import yaml
import configargparse
from dask.distributed import Client, LocalCluster
import dask.config

from .tools_xarray import construct_path


def configure(args: list, default_task_name: str = None) -> dict:
    """
    Process the given arguments, identify the configuration filepath, and extract the correct configuration settings for
    the requested module.

    Parameters
    ----------
    args : list
        List of input arguments
    default_task_name : str, optional
        Task name. If given, will return the specific configuration for the requested task. Otherwise return all
        configurations.

    Returns
    -------
    config : dict
        Dictionary with user configurations

    """

    commandline_args = process_commandline(args)

    path = commandline_args.configfile_path
    config = load_config(path)

    task_name = commandline_args.task
    if not task_name:
        task_name = default_task_name
    if task_name:
        config = extract_task_config(config, task_name)

    if commandline_args.cwd:
        config["cwd"] = commandline_args.cwd

    return config


def extract_task_config(config, task_name):
    if task_name is None:
        return config

    if "tasks" not in config:
        raise SystemError(
            f"No tasks section found in configuration file."
            "Cannot proceed to configure {task}"
        )

    if task_name not in config["tasks"]:
        raise SystemError(f"Task name {task_name} not found in configuration file")

    task_config = config["tasks"][task_name]["configuration"]
    for key, value in config.get("generic", {}).items():
        if key not in task_config:
            task_config[key] = value

    return task_config


def process_commandline(args):
    arg_parser = configargparse.ArgumentParser()
    arg_parser.add_argument(
        "configfile_path", type=str, help="The full path to the .json/.yaml config file"
    )
    arg_parser.add_argument(
        "--task", type=str, help="Name of task to perform", default=None
    )
    arg_parser.add_argument(
        "--cwd", type=str, help="Set current working directory", default=None
    )

    # deal with configuration
    if args is not None:
        commandline_args = arg_parser.parse_known_args(args=args)[0]
    else:
        raise SystemError("Need to provide a path to a configuration file")
    return commandline_args


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
    if not os.path.exists(path):
        raise SystemError(f"Configuration file {path} does not exist")

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


logging_config_defaults = {
    "format": "%(levelname)s:%(message)s",
    "level": logging.INFO,
}

dask_settings_default = {
    "config": {
        "distributed.scheduler.worker-ttl": None,
        "logging.distributed": "error",
    },
    "cluster": {
        "silence_logs": logging.ERROR,
    },
    "client": {},
    "chunk": {},
}


def preamble(args: list, task_name: str = None) -> tuple[dict, Client]:
    """
    Generic preamble for all modules.
    Read configuration file, set up dask client and logging.

    Parameters
    ----------
    args : list
        List of command line arguments

    Returns
    -------
    config : dict
        Dictionary with user configurations
    client : dask.distributed.Client
        Dask client
    """

    config = configure(args[1:], task_name)

    logging_config = config.get("logging", logging_config_defaults)
    logging.basicConfig(**logging_config)
    logging_config = config.get("logging", {})
    logging.basicConfig(**logging_config)
    logging.captureWarnings(True)

    use_dask = False
    dask_settings_from_config = {}
    if "chunk" in config:
        use_dask = True
    if "dask" in config:
        dask_check = config["dask"]
        if dask_check is False:
            use_dask = False
        else:
            dask_settings_from_config = dask_check

    if use_dask:
        dask_settings = dask_settings_default | dask_settings_from_config
        dask.config.set(dask_settings.get("config", {}))

        # for cluster and client we just use the combined settings
        config["__dask_cluster__"] = LocalCluster(**dask_settings["cluster"])
        config["__dask_client__"] = Client(
            config["__dask_cluster__"], **dask_settings["client"]
        )

    logging.info("dask status: %s", use_dask)

    if "cwd" in config:
        os.chdir(config["cwd"])

    return config


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def exchange_suffix(data_stores_in, suffix_map: None):
    data_stores = deepcopy(data_stores_in)
    for name, data in data_stores.items():
        data_stores[name] = _exchange_suffix(data, suffix_map)
    return data_stores


def _exchange_suffix(data, suffix_map: None):
    if suffix_map is None:
        suffix_map = {}
    if isinstance(data, collections.abc.Sequence):
        return [_exchange_suffix(d, suffix_map) for d in data]
    elif isinstance(data, collections.abc.Mapping):
        if "path" in data:
            path = construct_path(data["path"])
            new_suffix = suffix_map.get(path.suffix, path.suffix)
            data["path"] = path.with_suffix(new_suffix).as_posix()
        return data
