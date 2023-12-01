import os
import json
import yaml
import configargparse
from pathlib import Path


def configure(args, module_name=None):
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
                "Cannot proceed to configure {module_name}"
            )
        if module_name not in config["modules"]:
            raise SystemError(
                f"Module name {module_name} not found in configuration file"
            )
        config = config["modules"][module_name]

    return config


def load_config(path):
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
