import collections
import numpy as np
import xarray as xr
import datatree as dt
import warnings
from pathlib import Path
from copy import deepcopy


def assign_dims(dim_spec, xarray_ds=None):
    if xarray_ds is None:
        xarray_ds = xr.Dataset()

    dim_dict = {}
    for dim, spec in dim_spec.items():
        # if range_spec is a list return list, if range_spec is a dict, interpret dict
        if isinstance(spec, collections.abc.Sequence):
            dim_dict[dim] = spec

        elif isinstance(spec, collections.abc.Mapping):
            if "interval" in spec:
                dim_dict[dim] = range_from_dict(spec)
            else:
                dim_dict[dim] = range(spec.get("length", 0))

        else:
            raise SystemError(f"unknown dimension specification {spec}")

    new_ds = xarray_ds.expand_dims(dim_dict)

    for dim, spec in dim_spec.items():
        new_ds[dim].attrs.update(spec)

    return new_ds


def assign_coords(coord_spec, xarray_ds):
    coord_dict = {}
    attrs_dict = {}
    for coord, spec in coord_spec.items():
        # spec can only be a dict
        dim = spec.get("dim", None)
        if dim is None:
            raise SystemError(f"no dimension specified for coordinate {coord}")
        if dim not in xarray_ds.dims:
            raise SystemError(f"dimension {dim} not in xarray dataset")
        len = xarray_ds.dims[dim]
        if "values" in spec:
            range = spec["values"]
            if range.size != len:
                raise SystemError(
                    f"length of values for coordinate {coord} does not match dimension {dim}"
                )
            coord_dict[coord] = (dim, spec["values"])
        else:
            range = range_from_dict(spec, len=len)
            coord_dict[coord] = (dim, range)
        attrs_dict[coord] = spec

    new_ds = xarray_ds.assign_coords(coord_dict)
    for k, v in attrs_dict.items():
        new_ds[k].attrs.update(v)

    return new_ds


def range_from_dict(spec, len=None):
    interval = spec.get("interval")
    spacing = spec.get("sequence_spacing", "linear")
    multiplier = spec.get("multiplier", 1.0)
    offset = spec.get("offset", 0.0)
    if len is None:
        len = spec.get("length", 2)

    if spacing == "linear":
        range = np.linspace(interval[0], interval[1], len)
    elif spacing in ["log", "exp", "geometric"]:
        range = np.geomspace(interval[0], interval[1], len)
    elif spacing == "lin_values_to_log10":
        range = np.linspace(np.log10(interval[0]), np.log10(interval[1]), len)
    else:
        raise SystemError(f"unknown sequence spacing {spacing}")

    return multiplier * (offset + range)


def select_from_dims(filters, xarray_ds):
    selection = {}
    for k, v in filters.items():
        if k in xarray_ds.dims:
            selection[k] = v

    return selection


def chunk(xarray_ds, chunk_spec):
    chunks = select_from_dims(chunk_spec, xarray_ds)
    xarray_ds = xarray_ds.chunk(chunks)

    return xarray_ds


def filter(xarray_ds, filter_spec):
    filters = select_from_dims(filter_spec, xarray_ds)
    xarray_ds = xarray_ds.sel(filters)

    return xarray_ds


def thin(xarray_ds, thinning_spec):
    thinnings = select_from_dims(thinning_spec, xarray_ds)
    xarray_ds = xarray_ds.thin(thinnings)

    return xarray_ds


def construct_path(path_spec):
    if path_spec is None:
        return None
    if isinstance(path_spec, str):
        path = Path(path_spec)
    elif isinstance(path_spec, collections.abc.Sequence):
        path = Path().joinpath(*path_spec)

    return path


def data_source(**kwargs):
    kwargs_full = deepcopy(kwargs)
    type = kwargs.pop("type", None)
    if type is None:
        raise SystemError("No data source type specified")

    xarray_function = {
        "xarray_dataset": xr.open_dataset,
        "xarray_dataarray": xr.open_dataarray,
        "xarray_datatree": dt.open_datatree,
        "xarray_mfdataset": xr.open_mfdataset,  # also add {parallel : True} to the config
    }

    if type in xarray_function.keys():
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        path = construct_path(kwargs.pop("path"))
        if not "engine" in kwargs:
            if path.suffix == ".zarr":
                kwargs["engine"] = "zarr"
            else:
                kwargs["engine"] = "h5netcdf"
        source = xarray_function[type](path, **kwargs)
    else:
        raise SystemError(f"unknown data source type {type}")

    return source


def open(name, config, **kwargs):
    # within a module we open data sources
    # in a more generic context we allow opening any data store
    section_name = "data_sources"
    if section_name not in config:
        section_name = "data_stores"
    kwargs = make_group(config[section_name][name], kwargs)
    kwargs = config[section_name][name] | kwargs

    chunking_allowed = kwargs.pop("chunking_allowed", True)
    ds = (
        data_source(**kwargs)
        .pipe(filter, config.get("filters", {}))
        .pipe(thin, config.get("thinnings", {}))
    )
    # chunking with empty dict is not a non-op
    if "chunks" in config and chunking_allowed:
        ds = chunk(ds, config["chunks"])

    return ds


def open_tree(name, config, **kwargs):
    local_config = deepcopy(config)
    local_config["data_stores"][name]["type"] = "xarray_datatree"
    local_config["data_stores"][name].pop("group")
    ds = open(name, local_config, **kwargs)

    return ds


def store(ds, name, config, **kwargs):
    kwargs = make_group(config["data_sinks"][name], kwargs)
    kwargs = config["data_sinks"][name] | kwargs
    _ = kwargs.pop("type")
    path = construct_path(kwargs.pop("path"))

    if path.suffix == ".zarr":
        if not "consolidated" in kwargs:
            kwargs["consolidated"] = True
        if not "mode" in kwargs:
            kwargs["mode"] = "a"
        result = ds.drop_encoding().to_zarr(path, **kwargs)
    else:
        # Ensure .zarr arguments 'append_dim' asnd 'mode'='w-' are removed, as they are not technically possible to
        # use in .h5 files.
        # For 'append_dim' give a warning, because it only returns one iteration over the dimension.
        append_dimension = kwargs.pop("append_dim", False)
        if "mode" in kwargs and kwargs["mode"] == "w-":
            kwargs["mode"] = "w"
        if append_dimension is not False:
            warnings.warn(f"Storing data in .h5 format instead of intended .zarr format. "
                          f"Only one iteration over dimension {append_dimension} will be stored.",
                          category=UserWarning)
        if not "engine" in kwargs:
            kwargs["engine"] = "h5netcdf"
        result = ds.to_netcdf(path, **kwargs)

    return result


def make_group(config, kwargs):
    group = None
    if "group" in config:
        group = construct_path(config["group"])
    if "group" in kwargs:
        if group is None:
            group = construct_path(kwargs["group"])
        else:
            group = group / construct_path(kwargs["group"])
    if group is not None:
        kwargs["group"] = str(group)

    return kwargs


def prepare_ds(config):
    drop_vars = [d for d, v in config["dimensions"].items() if "interval" not in v]
    ds = assign_dims(config["dimensions"])
    if "coordinates" in config:
        ds = assign_coords(config["coordinates"], ds)
    ds = ds.drop_vars(drop_vars)

    return ds


def unpack(da):
    return da.values[()]
